"""
Source discovery & session configuration helpers.

Replaces the input-selection logic that previously lived in the terminal menu of
`run.py` (webcam / custom video / synthetic demo). Pure logic, no GUI imports.
"""

import os
import sys
import glob
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")
SAMPLE_VIDEO_DIR = "sample_videos"
DEMO_VIDEO_FILE = "demo_fall_with_occlusion.mp4"
DEMO_OUTPUT_FILE = "fall_detection_output.mp4"
WEBCAM_OUTPUT_FILE = "webcam_record.mp4"
MIN_VIDEO_BYTES = 1024

# Pose models offered to the user: (weights file, description). Ultralytics downloads
# weights that are not present yet on first use (internet connection required); if that
# fails the session falls back to the bundled model. Measured on home-webcam footage,
# yolo11s shows ~5x fewer ghost people than the nano model and finds fallen people more
# often, at ~1.6x the CPU time.
POSE_MODELS = [
    ("yolo11s-pose.pt", "Recommended - fewest ghosts"),
    ("yolov8n-pose.pt", "Fastest, works offline, more ghosts"),
    ("yolov8s-pose.pt", "Finds fallen people best, more ghosts"),
    ("yolov8m-pose.pt", "Largest - slow on CPU"),
]
DEFAULT_POSE_MODEL = "yolo11s-pose.pt"
BUNDLED_POSE_MODEL = "yolov8n-pose.pt"   # shipped with the repository

# Occlusion barrier drawn by the synthetic demo generator (coffee table)
DEMO_OCCLUSION_ZONES = [
    ("Coffee Table", [(300, 310), (620, 310), (620, 440), (300, 440)]),
]

# Generated outputs / artifacts that should not be offered as analysis inputs
_EXCLUDED_PREFIXES = ("output_", "test", "live_")
_EXCLUDED_FILES = {DEMO_VIDEO_FILE, DEMO_OUTPUT_FILE, WEBCAM_OUTPUT_FILE}


class SourceKind(str, Enum):
    WEBCAM = "webcam"
    VIDEO = "video"
    DEMO = "demo"


@dataclass
class VideoEntry:
    path: str
    size_kb: int
    is_benchmark: bool


@dataclass
class SessionConfig:
    kind: SourceKind
    source: str
    save_path: Optional[str]
    conf: float = 0.35
    model_name: str = DEFAULT_POSE_MODEL
    device: Optional[str] = None
    # Real capture rate of a video file when its header is wrong (e.g. MCFD clips are
    # 30 fps but declare 120). None = trust the file. Timing and speeds depend on it.
    fps_override: Optional[float] = None
    occlusion_zones: List[Tuple[str, List[Tuple[int, int]]]] = field(default_factory=list)

    @property
    def is_webcam(self) -> bool:
        return self.kind == SourceKind.WEBCAM


def discover_videos(root: str = ".") -> List[VideoEntry]:
    """Lists candidate input videos in the project root and `sample_videos/`."""
    candidates: List[str] = []
    for directory in (root, os.path.join(root, SAMPLE_VIDEO_DIR)):
        for ext in VIDEO_EXTENSIONS:
            candidates.extend(glob.glob(os.path.join(directory, f"*{ext}")))

    entries: List[VideoEntry] = []
    for path in sorted(set(candidates)):
        name = os.path.basename(path)
        if name.startswith(_EXCLUDED_PREFIXES) or name in _EXCLUDED_FILES:
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size <= MIN_VIDEO_BYTES:
            continue
        rel = os.path.relpath(path, root)
        entries.append(VideoEntry(
            path=rel,
            size_kb=size // 1024,
            is_benchmark=os.path.dirname(rel) == SAMPLE_VIDEO_DIR,
        ))

    # Benchmark clips first, then local files
    entries.sort(key=lambda e: (not e.is_benchmark, e.path))
    return entries


def validate_video_file(path: str) -> Optional[str]:
    """Returns an error message if the path is not a usable video file, else None."""
    clean = path.strip().strip('"').strip("'")
    if not clean:
        return "No video file selected."
    if not os.path.exists(clean):
        return f"File not found: '{clean}'"
    if os.path.getsize(clean) < MIN_VIDEO_BYTES:
        return f"File '{clean}' is empty or corrupt."
    return None


def open_with_system(path: str) -> None:
    """Opens a file or folder with the OS default application."""
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def output_path_for_video(path: str) -> str:
    base_name = os.path.splitext(os.path.basename(path))[0]
    return f"output_{base_name}.mp4"


def build_webcam_config(cam_index: int = 0, conf: float = 0.35, save: bool = True,
                        model_name: str = DEFAULT_POSE_MODEL) -> SessionConfig:
    return SessionConfig(
        kind=SourceKind.WEBCAM,
        source=str(int(cam_index)),
        save_path=WEBCAM_OUTPUT_FILE if save else None,
        conf=conf,
        model_name=model_name,
    )


def parse_fps_override(text: str) -> Optional[float]:
    """'Auto' / empty -> None, otherwise a frame rate between 1 and 240 (raises ValueError)."""
    clean = text.strip().lower()
    if not clean or clean.startswith("auto"):
        return None
    fps = float(clean)
    if not 1.0 <= fps <= 240.0:
        raise ValueError("Frame rate must be between 1 and 240.")
    return fps


def build_video_config(path: str, conf: float = 0.35, save: bool = True,
                       model_name: str = DEFAULT_POSE_MODEL,
                       fps_override: Optional[float] = None) -> SessionConfig:
    clean = path.strip().strip('"').strip("'")
    return SessionConfig(
        kind=SourceKind.VIDEO,
        source=clean,
        save_path=output_path_for_video(clean) if save else None,
        conf=conf,
        model_name=model_name,
        fps_override=fps_override,
    )


def build_demo_config(conf: float = 0.35, save: bool = True,
                      model_name: str = DEFAULT_POSE_MODEL) -> SessionConfig:
    return SessionConfig(
        kind=SourceKind.DEMO,
        source=DEMO_VIDEO_FILE,
        save_path=DEMO_OUTPUT_FILE if save else None,
        conf=conf,
        model_name=model_name,
        occlusion_zones=list(DEMO_OCCLUSION_ZONES),
    )
