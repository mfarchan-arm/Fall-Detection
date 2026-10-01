"""
Background detection session.

Runs capture -> FallDetectionPipeline -> video writer in a worker thread and
reports progress through a `SessionListener`. Contains no GUI code: the
listener callbacks are invoked from the worker thread, so a GUI front-end must
marshal them onto its own UI thread.
"""

import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np

from app.services import SessionConfig, SourceKind, BUNDLED_POSE_MODEL

ALERT_STATES = ("FALL_DETECTED", "IMPACT", "CONFIRMED_FALL")


@dataclass
class FrameUpdate:
    frame: np.ndarray          # Rendered BGR frame with HUD overlay
    frame_index: int
    total_frames: int          # 0 for live sources
    fps: float


@dataclass
class AlertEvent:
    frame_index: int
    track_id: int
    state: str
    torso_angle: float
    timestamp: float


@dataclass
class SessionSummary:
    frames: int
    duration_sec: float
    avg_fps: float
    save_path: Optional[str]
    stopped_by_user: bool
    error: Optional[str] = None


class SessionListener:
    """Callback interface. Override the hooks you need; all run on the worker thread."""

    def on_status(self, message: str) -> None: ...

    def on_frame(self, update: FrameUpdate) -> None: ...

    def on_alert(self, event: AlertEvent) -> None: ...

    def on_finished(self, summary: SessionSummary) -> None: ...


class _FileFrames:
    """Every frame of a video file, in order; timestamps are positions in the file."""

    def __init__(self, cap: cv2.VideoCapture, fps: float):
        self._cap = cap
        self._fps = fps
        self._index = 0

    def read(self) -> Optional[Tuple[np.ndarray, float]]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        timestamp = self._index / self._fps
        self._index += 1
        return frame, timestamp

    def close(self) -> None:
        pass


class _LiveFrames:
    """
    Grabs camera frames on its own thread and hands out only the newest one, so a
    detector slower than the camera analyses the present moment instead of working
    through a growing backlog of old frames. Timestamps are real capture times.
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._cond = threading.Condition()
        self._latest: Optional[Tuple[np.ndarray, float]] = None
        self._fresh = False
        self._alive = True
        self._thread = threading.Thread(target=self._grab_loop, name="CameraGrabber", daemon=True)
        self._thread.start()

    def _grab_loop(self) -> None:
        while self._alive:
            ok, frame = self._cap.read()
            timestamp = time.perf_counter()
            with self._cond:
                if not ok or frame is None:
                    self._alive = False
                else:
                    self._latest, self._fresh = (frame, timestamp), True
                self._cond.notify_all()

    def read(self, timeout: float = 5.0) -> Optional[Tuple[np.ndarray, float]]:
        with self._cond:
            self._cond.wait_for(lambda: self._fresh or not self._alive, timeout)
            if not self._fresh:
                return None
            self._fresh = False
            return self._latest

    def close(self) -> None:
        self._alive = False
        self._thread.join(timeout=2.0)


class _VideoSink:
    """
    Writes annotated frames. Live sources drop frames when detection is slower than the
    camera, so their output FPS is measured from the first seconds of processing to keep
    the recording at real-time speed.
    """
    MEASURE_SEC = 2.0

    def __init__(self, path: str, size: Tuple[int, int], fps: float, live: bool):
        self._path = path
        self._size = size
        self._fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._pending: List[np.ndarray] = []
        self._first_ts: Optional[float] = None
        self._last_ts = 0.0
        if not live:
            self._open(fps)

    def _open(self, fps: float) -> None:
        self._writer = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc(*"mp4v"), fps, self._size)

    def _flush_pending(self) -> None:
        elapsed = self._last_ts - (self._first_ts or 0.0)
        fps = (len(self._pending) - 1) / elapsed if elapsed > 0 and len(self._pending) > 1 else self._fps
        self._open(min(max(fps, 1.0), 60.0))
        for frame in self._pending:
            self._writer.write(frame)
        self._pending = []

    def write(self, frame: np.ndarray, timestamp: float) -> None:
        if self._writer is not None:
            self._writer.write(frame)
            return
        if self._first_ts is None:
            self._first_ts = timestamp
        self._last_ts = timestamp
        self._pending.append(frame)
        if timestamp - self._first_ts >= self.MEASURE_SEC:
            self._flush_pending()

    def close(self) -> None:
        if self._writer is None and self._pending:
            self._flush_pending()
        if self._writer is not None:
            self._writer.release()


class DetectionSession:
    def __init__(self, config: SessionConfig, listener: SessionListener):
        self.config = config
        self.listener = listener
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError("Session already running")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="DetectionSession", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        t_start = time.time()
        frame_count = 0
        error: Optional[str] = None
        try:
            frame_count = self._process()
        except Exception as e:  # Surface any failure to the front-end instead of dying silently
            error = str(e)
            self.listener.on_status(f"Error: {e}")

        total_time = time.time() - t_start
        self.listener.on_finished(SessionSummary(
            frames=frame_count,
            duration_sec=total_time,
            avg_fps=frame_count / max(total_time, 1e-4),
            save_path=os.path.abspath(self.config.save_path) if self.config.save_path and frame_count else None,
            stopped_by_user=self._stop_event.is_set(),
            error=error,
        ))

    def _process(self) -> int:
        # Heavy imports (torch / ultralytics) are deferred so the GUI can open instantly
        self.listener.on_status("Loading AI modules...")
        from fall_detection_all_in_one import FallDetectionPipeline, generate_fall_test_video

        cfg = self.config
        if cfg.kind == SourceKind.DEMO and not os.path.exists(cfg.source):
            self.listener.on_status("Generating synthetic fall & occlusion scenario...")
            generate_fall_test_video(cfg.source)

        self.listener.on_status(f"Opening source: {cfg.source}")
        if cfg.is_webcam and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(int(cfg.source), cv2.CAP_DSHOW)
        elif cfg.is_webcam:
            cap = cv2.VideoCapture(int(cfg.source))
        else:
            cap = cv2.VideoCapture(cfg.source)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source '{cfg.source}'. Check file path or camera permissions.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or fps > 240:
            fps = 30.0
        if cfg.fps_override and not cfg.is_webcam:
            fps = float(cfg.fps_override)
        if cfg.is_webcam:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frames = _LiveFrames(cap)
        else:
            frames = _FileFrames(cap, fps)

        sink: Optional[_VideoSink] = None
        frame_count = 0
        try:
            first = frames.read()
            if first is None:
                raise RuntimeError(f"Could not read frames from video source '{cfg.source}'")
            frame, timestamp = first

            height, width = frame.shape[:2]
            total_frames = 0 if cfg.is_webcam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if cfg.save_path:
                sink = _VideoSink(cfg.save_path, (width, height), fps, live=cfg.is_webcam)

            self.listener.on_status(f"Loading pose model ({cfg.model_name})...")
            try:
                pipeline = FallDetectionPipeline(model_name=cfg.model_name, conf_thresh=cfg.conf,
                                                 device=cfg.device, fps=fps)
            except Exception as e:
                if cfg.model_name == BUNDLED_POSE_MODEL:
                    raise
                # e.g. no internet to download the weights on first use
                self.listener.on_status(f"Could not load {cfg.model_name} ({e}); using {BUNDLED_POSE_MODEL}")
                pipeline = FallDetectionPipeline(model_name=BUNDLED_POSE_MODEL, conf_thresh=cfg.conf,
                                                 device=cfg.device, fps=fps)
            for name, polygon in cfg.occlusion_zones:
                pipeline.add_furniture_zone(name, polygon)

            self.listener.on_status(f"Running on {cfg.source} ({width}x{height} @ {fps:.1f} FPS)")

            # Alert only on state transitions so a fall isn't reported every frame
            last_state: Dict[int, str] = {}

            while not self._stop_event.is_set():
                if frame is None:
                    item = frames.read()
                    if item is None:
                        break
                    frame, timestamp = item

                frame_count += 1
                t_frame = time.time()
                vis_frame, alerts = pipeline.process_frame(frame, timestamp)
                current_fps = 1.0 / max(time.time() - t_frame, 1e-4)
                pipeline.fps = current_fps

                self._emit_alerts(alerts, last_state, frame_count)

                if sink:
                    sink.write(vis_frame, timestamp)

                self.listener.on_frame(FrameUpdate(
                    frame=vis_frame,
                    frame_index=frame_count,
                    total_frames=total_frames,
                    fps=current_fps,
                ))
                frame = None
        finally:
            frames.close()
            cap.release()
            if sink:
                sink.close()

        return frame_count

    def _emit_alerts(self, alerts: list, last_state: Dict[int, str], frame_index: int) -> None:
        active: Dict[int, Dict[str, Any]] = {a["track_id"]: a for a in alerts if a["state"] in ALERT_STATES}

        for track_id, alert in active.items():
            if last_state.get(track_id) == alert["state"]:
                continue
            self.listener.on_alert(AlertEvent(
                frame_index=frame_index,
                track_id=track_id,
                state=alert["state"],
                torso_angle=float(alert["features"].get("torso_angle", 0.0)),
                timestamp=time.time(),
            ))

        last_state.clear()
        last_state.update({t: a["state"] for t, a in active.items()})
