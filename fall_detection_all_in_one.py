"""
========================================================================================
  AI PERSON & ELDERLY FALL DETECTION SYSTEM (ALL-IN-ONE STANDALONE SCRIPT)
  Features:
    - Real-Time 17-Keypoint Pose Estimation (YOLOv8-Pose)
    - Advanced Occlusion Handling & Kinematic Bone Imputation
    - Temporal Trajectory & Furniture ROI Zone Tracking
    - Biomechanical Feature Extraction (Torso Angle, Vertical Velocity & Acceleration)
    - 6-Stage Fall State Machine (ADL False Alarm Suppression)
    - High-Contrast Telemetry HUD Visualizer
    - Built-in Synthetic Occlusion Scenario Generator & Test Runner
========================================================================================
"""

import os
import sys
import time
import bisect
import argparse
from collections import deque
from typing import List, Dict, Tuple, Optional, Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO


# States that represent an active fall alarm (IMPACT is kept for API compatibility)
ALERT_STATES = ("FALL_DETECTED", "IMPACT", "CONFIRMED_FALL")


# ======================================================================================
# 1. POSE DETECTOR MODULE
# ======================================================================================
class PoseDetector:
    """
    Wraps YOLOv8-Pose model for efficient single/multi-person landmark estimation.
    """
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    def __init__(
        self,
        model_name: str = "yolov8n-pose.pt",
        conf_thresh: float = 0.35,
        device: Optional[str] = None,
        min_joints: int = 3,
        joint_conf: float = 0.25,
        track_conf: float = 0.15
    ):
        self.conf_thresh = conf_thresh
        # Detections between track_conf and conf_thresh are "weak": they may only extend an
        # existing track (a person lying on the floor often scores low), never create one.
        self.track_conf = min(track_conf, conf_thresh)
        self.min_joints = min_joints
        self.joint_conf = joint_conf
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PoseDetector] Loading {model_name} on device: {self.device}")
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame,
            conf=self.track_conf,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kps_data = result.keypoints.data.cpu().numpy()

        for i in range(len(boxes)):
            bbox = boxes[i].tolist()
            det_conf = float(confs[i])
            kps = kps_data[i]

            if kps.shape[-1] == 2:
                kps_with_conf = np.zeros((17, 3), dtype=np.float32)
                kps_with_conf[:, :2] = kps
                kps_with_conf[:, 2] = np.where((kps[:, 0] > 0) | (kps[:, 1] > 0), det_conf, 0.0)
                kps = kps_with_conf

            # Reject tiny boxes and near-empty skeletons. Deliberately permissive: lying /
            # fallen bodies get low keypoint confidence and must not be dropped right when
            # the fall happens.
            confident_joints = np.sum((kps[:, 2] > self.joint_conf) & ((kps[:, 0] > 0) | (kps[:, 1] > 0)))
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            if confident_joints < self.min_joints or bbox_w < 20 or bbox_h < 30:
                continue

            detections.append({
                "bbox": bbox,
                "det_conf": det_conf,
                "keypoints": kps,
                "strong": det_conf >= self.conf_thresh
            })

        return detections


# ======================================================================================
# 2. OCCLUSION & KINEMATIC BONE IMPUTATION ENGINE
# ======================================================================================
class OcclusionEngine:
    """
    Recovers occluded joints through kinematic bone constraints, temporal momentum,
    and furniture zone tracking. Imputed joints are used for display and occlusion
    statistics only; fall decisions are made from observed joints.
    """
    SKELETON_PAIRS = [
        (0, 1), (0, 2), (1, 3), (2, 4),            # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Arms
        (5, 11), (6, 12), (11, 12),                # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)     # Legs
    ]

    UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    TORSO_INDICES = [5, 6, 11, 12]
    LOWER_BODY_INDICES = [13, 14, 15, 16]

    def __init__(self, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.occlusion_zones: List[Dict[str, Any]] = []

    def add_occlusion_zone(self, name: str, polygon: List[Tuple[int, int]], zone_type: str = "furniture"):
        self.occlusion_zones.append({
            "name": name,
            "polygon": np.array(polygon, dtype=np.int32),
            "type": zone_type
        })

    def process_keypoints(
        self,
        current_kps: np.ndarray,
        history_kps: List[np.ndarray],
        bbox: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        history_kps: imputed keypoints of the PREVIOUS frames (current frame excluded).
        """
        imputed_kps = np.copy(current_kps)
        occlusion_mask = np.zeros(17, dtype=bool)

        for i in range(17):
            conf = current_kps[i, 2]
            x, y = current_kps[i, 0], current_kps[i, 1]
            if conf < self.conf_threshold or (x == 0 and y == 0):
                occlusion_mask[i] = True

        total_kps = 17
        visible_total = np.sum(~occlusion_mask)
        upper_vis = np.mean(~occlusion_mask[self.UPPER_BODY_INDICES])
        lower_vis = np.mean(~occlusion_mask[self.LOWER_BODY_INDICES])
        torso_vis = np.mean(~occlusion_mask[self.TORSO_INDICES])

        # Temporal Momentum Extrapolation
        if len(history_kps) >= 2:
            prev_kps = history_kps[-1]
            prev_prev_kps = history_kps[-2]

            for i in range(17):
                if occlusion_mask[i]:
                    if prev_kps[i, 0] > 0 and prev_prev_kps[i, 0] > 0:
                        vx = (prev_kps[i, 0] - prev_prev_kps[i, 0]) * 0.9
                        vy = (prev_kps[i, 1] - prev_prev_kps[i, 1]) * 0.9
                        imputed_kps[i, 0] = prev_kps[i, 0] + vx
                        imputed_kps[i, 1] = prev_kps[i, 1] + vy
                        imputed_kps[i, 2] = 0.45

        # Biomechanical Bone Constraint Imputation for Lower Body
        bbox_h = bbox[3] - bbox[1]
        if occlusion_mask[13] and not occlusion_mask[11]:  # Left Knee
            imputed_kps[13, 0] = imputed_kps[11, 0]
            imputed_kps[13, 1] = imputed_kps[11, 1] + bbox_h * 0.3
            imputed_kps[13, 2] = 0.35

        if occlusion_mask[14] and not occlusion_mask[12]:  # Right Knee
            imputed_kps[14, 0] = imputed_kps[12, 0]
            imputed_kps[14, 1] = imputed_kps[12, 1] + bbox_h * 0.3
            imputed_kps[14, 2] = 0.35

        if occlusion_mask[15] and imputed_kps[13, 1] > 0:  # Left Ankle
            imputed_kps[15, 0] = imputed_kps[13, 0]
            imputed_kps[15, 1] = imputed_kps[13, 1] + bbox_h * 0.25
            imputed_kps[15, 2] = 0.35

        if occlusion_mask[16] and imputed_kps[14, 1] > 0:  # Right Ankle
            imputed_kps[16, 0] = imputed_kps[14, 0]
            imputed_kps[16, 1] = imputed_kps[14, 1] + bbox_h * 0.25
            imputed_kps[16, 2] = 0.35

        # Keep guessed joints near the body: momentum extrapolation must not drift them away
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        guessed = occlusion_mask & (imputed_kps[:, 0] != 0)
        imputed_kps[guessed, 0] = np.clip(imputed_kps[guessed, 0], bbox[0] - 0.2 * bw, bbox[2] + 0.2 * bw)
        imputed_kps[guessed, 1] = np.clip(imputed_kps[guessed, 1], bbox[1] - 0.2 * bh, bbox[3] + 0.2 * bh)

        stats = {
            "total_visibility": float(visible_total / total_kps),
            "upper_body_visibility": float(upper_vis),
            "lower_body_visibility": float(lower_vis),
            "torso_visibility": float(torso_vis),
            "is_partially_occluded": bool(visible_total < total_kps and visible_total >= 5),
            "is_severely_occluded": bool(visible_total < 5)
        }

        return imputed_kps, occlusion_mask, stats

    def is_in_occlusion_zone(self, point: Tuple[float, float]) -> Tuple[bool, Optional[str]]:
        pt = (int(point[0]), int(point[1]))
        for zone in self.occlusion_zones:
            dist = cv2.pointPolygonTest(zone["polygon"], pt, measureDist=False)
            if dist >= 0:
                return True, zone["name"]
        return False, None


# ======================================================================================
# 3. KINEMATIC & BIOMECHANICAL FEATURE EXTRACTOR
# ======================================================================================
class KinematicExtractor:
    """
    Computes torso angle, normalized vertical / angular velocity and motion energy.

    Robustness rules (these keep keypoint jitter from looking like a fall):
      * Features come from RAW detections only. Joints below `kp_conf` are ignored instead
        of being read as (0, 0), and imputed joints never drive a fall decision.
      * Velocities compare the SAME joints in two frames ~`velocity_window_sec` apart, using
        real timestamps and the median over joints, so one noisy or missing joint cannot
        spike them and dropped webcam frames cannot inflate them.
      * Everything is normalized by the person's upright reference height, so a body that
        shrinks while bending or falling does not inflate its own speed.
      * The torso angle is only trusted when the torso is long enough in the image; a
        foreshortened torso (facing / bending toward the camera) gives angle_valid=False.
      * Joints on the image border are ignored: the detector parks body parts that are
        outside the picture there, so they are not real positions.
    """
    MOTION_JOINTS = [0, 1, 2, 3, 4, 5, 6, 11, 12]  # head, shoulders, hips
    UPPER_JOINTS = [0, 1, 2, 3, 4, 5, 6]           # head, shoulders

    def __init__(
        self,
        kp_conf: float = 0.3,
        velocity_window_sec: float = 0.25,
        motion_window_sec: float = 0.6,
        min_torso_ratio: float = 0.12,
        angle_smoothing: float = 0.5
    ):
        self.kp_conf = kp_conf
        self.velocity_window_sec = velocity_window_sec
        self.motion_window_sec = motion_window_sec
        self.min_torso_ratio = min_torso_ratio
        self.angle_smoothing = angle_smoothing
        self.frame_size: Optional[Tuple[int, int]] = None  # (width, height), set by the pipeline

    def extract_features(self, track: "Tracklet") -> Dict[str, Any]:
        t = track.time_history[-1]
        kps = track.raw_kps_history[-1]
        bbox = track.bbox_history[-1]
        bbox_w = max(bbox[2] - bbox[0], 1.0)
        bbox_h = max(bbox[3] - bbox[1], 1.0)
        ref_h = max(track.ref_height, 1.0)

        # Torso angle, EMA-smoothed across consecutive valid measurements
        raw_angle = self.torso_angle(kps, ref_h)
        shoulder, hip = self._mean_point(kps, [5, 6]), self._mean_point(kps, [11, 12])
        torso_ratio = float(np.hypot(*(shoulder - hip)) / ref_h) if shoulder is not None and hip is not None else 0.0
        # The angle ignores direction; a body lying head-toward-the-camera also reads ~90 deg,
        # so "upright" additionally needs the shoulders above the hips
        head_up = bool(shoulder is not None and hip is not None and shoulder[1] < hip[1])
        prev_angle = track.angle_history[-2] if len(track.angle_history) >= 2 else None
        if raw_angle is None or prev_angle is None:
            angle = raw_angle
        else:
            angle = prev_angle + self.angle_smoothing * (raw_angle - prev_angle)
        track.angle_history[-1] = angle
        if angle is not None:
            track.last_valid_angle = angle

        hip_y = self._mean_coord(kps, [11, 12], axis=1)
        track.hip_history[-1] = hip_y
        upper_y = self._mean_coord(kps, self.UPPER_JOINTS, axis=1)
        track.upper_history[-1] = upper_y
        # A box touching a border is only "cut" if the body part beyond that border is missing
        # (a person lying along the bottom of the picture with visible legs is not cut)
        cut_bottom = track.edges["bottom"] and not np.any(self._valid(kps, [13, 14, 15, 16]))
        cut_top = track.edges["top"] and not np.any(self._valid(kps, [0, 1, 2, 3, 4]))

        v_y = 0.0
        angular_vel = 0.0
        k = self._reference_index(track.time_history, t - self.velocity_window_sec)
        if k is not None:
            dt = t - track.time_history[k]
            dy = self._median_joint_shift(track.raw_kps_history[k], kps)
            if dy is not None:
                v_y = dy / (dt * ref_h)
            prev = track.angle_history[k]
            if angle is not None and prev is not None:
                angular_vel = (prev - angle) / dt

        return {
            "torso_angle": float(angle if angle is not None else track.last_valid_angle),
            "angle_valid": angle is not None,
            "torso_ratio": torso_ratio,             # torso length / upright height
            "head_up": head_up,                     # shoulders above hips in the image
            "angular_velocity": float(angular_vel),
            "vertical_velocity": float(v_y),
            "aspect_ratio": float(bbox_w / bbox_h),
            "height_ratio": float(bbox_h / ref_h),
            "stillness_energy": float(self._motion_energy(track, ref_h)),
            "centroid_y": float((bbox[1] + bbox[3]) / 2.0),
            "top_y": float(bbox[1]),
            "hip_y": hip_y,
            "upper_y": upper_y,
            "truncated": cut_bottom or cut_top,      # box height / shape distorted by a border
            "truncated_top": track.edges["top"],
            "bbox_height": float(bbox_h),
            "bbox_width": float(bbox_w),
            "ref_height": float(ref_h)
        }

    def torso_angle(self, kps: np.ndarray, ref_h: float) -> Optional[float]:
        """Torso angle vs. the floor (90 = upright, 0 = flat), or None if unreliable."""
        shoulder = self._mean_point(kps, [5, 6])
        hip = self._mean_point(kps, [11, 12])
        if shoulder is None or hip is None:
            return None
        dx, dy = shoulder - hip
        if np.hypot(dx, dy) < self.min_torso_ratio * ref_h:
            return None
        return float(np.degrees(np.arctan2(abs(dy), abs(dx))))

    def _valid(self, kps: np.ndarray, idx: List[int]) -> np.ndarray:
        sub = kps[idx]
        valid = (sub[:, 2] >= self.kp_conf) & ((sub[:, 0] > 0) | (sub[:, 1] > 0))
        if self.frame_size is not None:
            w, h = self.frame_size
            m = edge_margin(self.frame_size)
            valid &= (sub[:, 0] > m) & (sub[:, 0] < w - m) & (sub[:, 1] > m) & (sub[:, 1] < h - m)
        return valid

    def _mean_point(self, kps: np.ndarray, idx: List[int]) -> Optional[np.ndarray]:
        valid = self._valid(kps, idx)
        if not np.any(valid):
            return None
        return kps[idx][valid, :2].mean(axis=0)

    def _mean_coord(self, kps: np.ndarray, idx: List[int], axis: int) -> Optional[float]:
        point = self._mean_point(kps, idx)
        return None if point is None else float(point[axis])

    def _median_joint_shift(self, kps_a: np.ndarray, kps_b: np.ndarray) -> Optional[float]:
        """Median vertical shift (px, + = down) of joints visible in both frames."""
        idx = self.MOTION_JOINTS
        both = self._valid(kps_a, idx) & self._valid(kps_b, idx)
        if np.count_nonzero(both) < 2:
            return None
        return float(np.median(kps_b[idx][both, 1] - kps_a[idx][both, 1]))

    def _median_joint_distance(self, kps_a: np.ndarray, kps_b: np.ndarray) -> Optional[float]:
        idx = self.MOTION_JOINTS
        both = self._valid(kps_a, idx) & self._valid(kps_b, idx)
        if np.count_nonzero(both) < 2:
            return None
        return float(np.median(np.linalg.norm(kps_b[idx][both, :2] - kps_a[idx][both, :2], axis=1)))

    def _reference_index(self, times: List[float], target: float) -> Optional[int]:
        """Latest earlier frame at or before `target`, or the oldest one if history is short."""
        if len(times) < 2:
            return None
        k = bisect.bisect_right(times, target) - 1
        if k < 0:
            if times[-1] - times[0] < 0.5 * self.velocity_window_sec:
                return None
            k = 0
        k = min(k, len(times) - 2)
        return k if times[-1] - times[k] > 1e-6 else None

    def _motion_energy(self, track: "Tracklet", ref_h: float) -> float:
        """Mean joint speed over the recent window, in body heights per second."""
        times = track.time_history
        t_end = times[-1]
        total_shift, total_dt = 0.0, 0.0
        for i in range(len(times) - 1, 0, -1):
            if times[i - 1] < t_end - self.motion_window_sec:
                break
            d = self._median_joint_distance(track.raw_kps_history[i - 1], track.raw_kps_history[i])
            if d is None:
                continue
            total_shift += d
            total_dt += times[i] - times[i - 1]
        return total_shift / (total_dt * ref_h) if total_dt > 0 else 0.0


# ======================================================================================
# 4. TRACKLET & MULTI-OBJECT TRACKER
# ======================================================================================
NO_EDGES = {"left": False, "top": False, "right": False, "bottom": False}


def edge_margin(frame_size: Optional[Tuple[int, int]]) -> float:
    """Pixels from the image border within which a box or joint counts as clipped."""
    if frame_size is None:
        return 0.0
    return max(3.0, 0.01 * max(frame_size))


def touching_edges(bbox: List[float], frame_size: Optional[Tuple[int, int]]) -> Dict[str, bool]:
    """Image borders that clip a box: a clipped box no longer shows the whole body."""
    if frame_size is None:
        return dict(NO_EDGES)
    w, h = frame_size
    m = edge_margin(frame_size)
    return {"left": bbox[0] <= m, "top": bbox[1] <= m, "right": bbox[2] >= w - m, "bottom": bbox[3] >= h - m}


class Tracklet:
    HISTORY_SEC = 3.0

    def __init__(self, track_id: int, initial_bbox: List[float], initial_kps: np.ndarray,
                 timestamp: float = 0.0, det_conf: float = 1.0, edges: Optional[Dict[str, bool]] = None):
        self.track_id = track_id
        # Per-frame histories, all index-aligned
        self.bbox_history: List[List[float]] = []
        self.raw_kps_history: List[np.ndarray] = []
        self.imputed_kps_history: List[np.ndarray] = []
        self.time_history: List[float] = []
        self.angle_history: List[Optional[float]] = []
        self.hip_history: List[Optional[float]] = []
        self.upper_history: List[Optional[float]] = []
        self.top_clipped_history: List[bool] = []
        self.missing_frames: int = 0
        self.last_seen: float = timestamp
        self.first_seen: float = timestamp
        self.hits: int = 0                        # strong detections so far
        self.max_conf: float = 0.0
        self.conf_ema: Optional[float] = None     # recency-weighted confidence of strong hits
        self.confirmed: bool = False              # tentative until it repeatedly looked like a person
        self.edges: Dict[str, bool] = dict(NO_EDGES)
        self.in_occlusion_zone: bool = False
        self.occlusion_zone_name: Optional[str] = None
        self.state: str = "MONITORING"
        self.state_timer: float = 0.0
        self.seen_down: bool = False              # last observed posture was down on the floor
        self.ref_height: float = max(initial_bbox[3] - initial_bbox[1], 1.0)
        self.last_valid_angle: float = 90.0
        self.fsm: Dict[str, Any] = {}             # scratch values owned by FallStateMachine
        self.last_data: Optional[Dict[str, Any]] = None
        self.update(initial_bbox, initial_kps, timestamp, det_conf, True, edges)

    @property
    def is_visible(self) -> bool:
        return self.missing_frames == 0

    @property
    def at_edge(self) -> bool:
        return any(self.edges.values())

    def update(self, bbox: List[float], raw_kps: np.ndarray, timestamp: float, det_conf: float = 1.0,
               strong: bool = True, edges: Optional[Dict[str, bool]] = None):
        self.bbox_history.append(bbox)
        self.raw_kps_history.append(raw_kps)
        self.imputed_kps_history.append(np.copy(raw_kps))  # replaced by the occlusion engine
        self.time_history.append(timestamp)
        self.angle_history.append(None)                     # filled by the kinematic extractor
        self.hip_history.append(None)
        self.upper_history.append(None)
        self.missing_frames = 0
        self.last_seen = timestamp
        if strong:
            self.hits += 1
            self.max_conf = max(self.max_conf, det_conf)
            self.conf_ema = det_conf if self.conf_ema is None else 0.6 * self.conf_ema + 0.4 * det_conf
        if edges is not None:
            self.edges = edges
        self.top_clipped_history.append(self.edges["top"])
        while len(self.time_history) > 2 and self.time_history[0] < timestamp - self.HISTORY_SEC:
            for history in (self.bbox_history, self.raw_kps_history, self.imputed_kps_history, self.time_history,
                            self.angle_history, self.hip_history, self.upper_history, self.top_clipped_history):
                history.pop(0)

    def mark_missing(self):
        self.missing_frames += 1


class SimpleTracker:
    """
    Greedy tracker (IoU OR normalized center distance) with a track life cycle:

      * tentative -> confirmed after `confirm_hits` strong detections whose recency-weighted
        confidence reaches `confirm_conf`. Ghosts (a chair, a coat, a shadow) rarely score that
        high consistently; a tentative track is never drawn and can never raise a fall.
      * Weak (low-confidence) detections may only extend a CONFIRMED track they overlap well
        (IoU >= weak_min_iou), e.g. a person lying on the floor who scores low. A match on
        distance alone needs a confident box (jump_conf), so a ghost cannot steal a track.
      * While a fall is in progress, a body-sized box where the person must have landed
        re-attaches to the track (the detector often loses a body as it reaches the floor).
      * A track lost at a frame border walked out of view and is dropped almost at once.
        A track lost in an alarm state is kept (the person is down but not visible), and a
        person standing up where it lay is re-attached to it so the alarm can clear.
    """

    def __init__(
        self,
        iou_thresh: float = 0.2,
        max_center_dist: float = 0.75,
        confirm_hits: int = 3,
        confirm_conf: float = 0.6,
        weak_min_iou: float = 0.3,
        jump_conf: float = 0.6,
        tentative_missing_sec: float = 0.3,
        max_missing_sec: float = 1.0,
        exit_missing_sec: float = 0.3,
        alert_missing_sec: float = 30.0,
        event_missing_sec: float = 2.0,
        merge_window_sec: float = 1.5
    ):
        self.iou_thresh = iou_thresh
        self.max_center_dist = max_center_dist
        self.confirm_hits = confirm_hits
        self.confirm_conf = confirm_conf
        self.weak_min_iou = weak_min_iou
        self.jump_conf = jump_conf
        self.tentative_missing_sec = tentative_missing_sec
        self.max_missing_sec = max_missing_sec
        self.exit_missing_sec = exit_missing_sec
        self.alert_missing_sec = alert_missing_sec
        self.event_missing_sec = event_missing_sec
        self.merge_window_sec = merge_window_sec
        self.tracklets: Dict[int, Tracklet] = {}
        self.next_id: int = 1

    def update(self, detections: List[Dict[str, Any]], timestamp: float,
               frame_size: Optional[Tuple[int, int]] = None) -> List[Tracklet]:
        for det in detections:
            det["edges"] = touching_edges(det["bbox"], frame_size)
        tracks = list(self.tracklets.values())
        matched_tracks: set = set()
        matched_dets = self._associate(tracks, detections, matched_tracks, timestamp)

        for ti, track in enumerate(tracks):
            if ti in matched_tracks:
                # Confirm on a sustained confidence, not one lucky frame: shadows and hanging
                # clothes occasionally score high but mostly score low
                if (not track.confirmed and track.hits >= self.confirm_hits
                        and (track.conf_ema or 0.0) >= self.confirm_conf):
                    track.confirmed = True
                continue
            track.mark_missing()
        self._adopt_split_tracks(timestamp)
        for track in list(self.tracklets.values()):
            if not track.is_visible and timestamp - track.last_seen > self._missing_limit(track):
                del self.tracklets[track.track_id]

        # A leftover detection lying mostly inside someone already tracked this frame is a
        # duplicate (e.g. an extra upper-body box), not a new person
        occupied = [tracks[ti].bbox_history[-1] for ti in matched_tracks]
        for di, det in enumerate(detections):
            if di in matched_dets or not det.get("strong", True):
                continue
            if any(self._containment(det["bbox"], box) > 0.6 for box in occupied):
                continue
            self.tracklets[self.next_id] = Tracklet(self.next_id, det["bbox"], det["keypoints"], timestamp,
                                                    det.get("det_conf", 1.0), det["edges"])
            self.next_id += 1

        return list(self.tracklets.values())

    @staticmethod
    def _landing_box(track: Tracklet, det: Dict[str, Any]) -> bool:
        """
        Could `det` be the falling person after landing? It must be body-sized (a bag or a
        chair leg is not), lie lower than the person was, reach down to where they stood,
        and be within a body length sideways.
        """
        if det.get("det_conf", 1.0) < 0.2:
            return False
        last, box, h0 = track.bbox_history[-1], det["bbox"], max(track.ref_height, 1.0)
        if np.hypot(box[2] - box[0], box[3] - box[1]) < 0.6 * h0:
            return False
        if box[1] < last[1] - 0.1 * h0 or box[3] < last[3] - 0.3 * h0:
            return False
        cx = (box[0] + box[2]) / 2.0
        return last[0] - h0 <= cx <= last[2] + h0

    def _adopt_split_tracks(self, timestamp: float):
        """
        While someone falls, the detector sometimes re-detects them as a separate box, so the
        fall history stays with a track that is now lost while a young track carries on. Hand
        the young track's latest observation back to the track that holds the fall.
        """
        tracks = list(self.tracklets.values())
        in_event = [t for t in tracks if t.confirmed and not t.is_visible and t.state != "MONITORING"]
        young = [t for t in tracks if t.is_visible and t.confirmed and t.state == "MONITORING"
                 and timestamp - t.first_seen <= self.merge_window_sec]
        for old in in_event:
            box = old.bbox_history[-1]
            near = [t for t in young if self._near(box, t.bbox_history[-1])]
            if not near:
                continue
            new = max(near, key=lambda t: self._calculate_iou(box, t.bbox_history[-1]))
            old.update(new.bbox_history[-1], new.raw_kps_history[-1], timestamp, new.max_conf, True, new.edges)
            young.remove(new)
            del self.tracklets[new.track_id]

    def _near(self, box: List[float], other: List[float]) -> bool:
        scale = max(box[2] - box[0], box[3] - box[1], 1.0)
        return (self._calculate_iou(box, other) >= 0.05 or self._center_distance(box, other) / scale <= 1.0
                or self._stands_up_from(box, other))

    def _missing_limit(self, track: Tracklet) -> float:
        if not track.confirmed:
            return self.tentative_missing_sec
        if track.at_edge and not track.seen_down:
            return self.exit_missing_sec          # walked out of view
        if track.state in ALERT_STATES:
            return self.alert_missing_sec         # down and not visible: keep the alarm
        if track.state != "MONITORING":
            return self.event_missing_sec         # fall in progress: the detector often loses
        return self.max_missing_sec               # a body right as it reaches the floor

    def _associate(self, tracks: List[Tracklet], detections: List[Dict[str, Any]],
                   matched_tracks: set, timestamp: float) -> set:
        """
        Greedy best-geometry matching of tracks to detections. Strong and weak detections
        compete in one pass, so a well-overlapping weak box (a person lying on the floor)
        beats a strong box that is merely nearby (a ghost next to them). Confidence only
        decides eligibility. Returns matched detection indices.
        """
        candidates = []
        for ti, track in enumerate(tracks):
            if ti in matched_tracks:
                continue
            last_bbox = track.bbox_history[-1]
            scale = max(last_bbox[2] - last_bbox[0], last_bbox[3] - last_bbox[1], 1.0)
            lost_in_alarm = track.missing_frames > 0 and track.state in ALERT_STATES
            falling = track.confirmed and track.state == "RAPID_DESCENT"
            for di, det in enumerate(detections):
                iou = self._calculate_iou(last_bbox, det["bbox"])
                dist = self._center_distance(last_bbox, det["bbox"]) / scale
                score = iou + max(0.0, 1.0 - dist / self.max_center_dist)
                strong = det.get("strong", True)
                if strong:
                    # A match on distance alone (boxes barely overlapping) must be a confident
                    # detection: otherwise a ghost next to a person can steal their track
                    normal = iou >= self.iou_thresh or (dist <= self.max_center_dist
                                                        and det.get("det_conf", 1.0) >= self.jump_conf)
                else:
                    # Weak boxes may only extend a confirmed track they overlap well
                    normal = track.confirmed and iou >= self.weak_min_iou
                if normal:
                    candidates.append((score, ti, di))
                elif falling and self._landing_box(track, det):
                    # Re-find a falling person where they must have landed; low priority
                    candidates.append((0.02 + iou, ti, di))
                elif strong and lost_in_alarm and self._stands_up_from(last_bbox, det["bbox"]):
                    # The fallen person getting up where they lay; lowest priority, so anyone
                    # else's closer match wins
                    candidates.append((0.01, ti, di))
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_dets = set()
        for _, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            det = detections[di]
            tracks[ti].update(det["bbox"], det["keypoints"], timestamp, det.get("det_conf", 1.0),
                              det.get("strong", True), det.get("edges"))
        return matched_dets

    @staticmethod
    def _stands_up_from(lying_box: List[float], box: List[float]) -> bool:
        """Is `box` centered where a person lying in `lying_box` would be after standing up?"""
        lw, lh = lying_box[2] - lying_box[0], lying_box[3] - lying_box[1]
        size = max(lw, lh)
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        return (lying_box[0] - 0.25 * lw <= cx <= lying_box[2] + 0.25 * lw
                and lying_box[1] - 1.5 * size <= cy <= lying_box[3] + 0.25 * lh)

    @staticmethod
    def _containment(inner: List[float], outer: List[float]) -> float:
        """Fraction of `inner`'s area that lies inside `outer`."""
        w = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
        h = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
        area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-6)
        return float(w * h / area)

    @staticmethod
    def _center_distance(boxA: List[float], boxB: List[float]) -> float:
        ax, ay = (boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0
        bx, by = (boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0
        return float(np.hypot(ax - bx, ay - by))

    def _calculate_iou(self, boxA: List[float], boxB: List[float]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

        return float(interArea / float(boxAArea + boxBArea - interArea + 1e-6))


# ======================================================================================
# 5. MULTI-STAGE FALL STATE MACHINE
# ======================================================================================
class FallStateMachine:
    """
    A fall is a FAST descent that ENDS with the body down on the ground:

        MONITORING    --(fast descent sustained >= trigger_min_sec)--> RAPID_DESCENT  (no alarm yet)
        RAPID_DESCENT --(body down for >= down_min_sec)--------------> FALL_DETECTED  (alarm)
        RAPID_DESCENT --(not down within confirm_window_sec)---------> MONITORING     (bend/sit/squat)
        FALL_DETECTED --(still not up after inactivity_sec)----------> CONFIRMED_FALL (emergency)
        FALL_DETECTED / CONFIRMED_FALL --(upright for recovery_sec)--> MONITORING

    "Down" is measured against the posture right before the descent: the top of the body
    must have dropped and, when hips are visible, the hips must have dropped too (bending
    over keeps the hips up; lying down on a bed barely lowers them).
    """
    STATE_MONITORING = "MONITORING"
    STATE_RAPID_DESCENT = "RAPID_DESCENT"
    STATE_FALL_DETECTED = "FALL_DETECTED"
    STATE_IMPACT = "IMPACT"
    STATE_INACTIVITY_CHECK = "INACTIVITY_CHECK"
    STATE_PRE_ALERT = "PRE_ALERT"
    STATE_CONFIRMED_FALL = "CONFIRMED_FALL"

    def __init__(
        self,
        descent_velocity_thresh: float = 0.7,   # body heights / s, measured over ~0.25 s
        collapse_rate_thresh: float = 120.0,    # deg / s of torso rotation toward the floor
        collapse_min_velocity: float = 0.3,     # ...only counts while also descending
        trigger_min_sec: float = 0.1,
        confirm_window_sec: float = 2.0,
        down_min_sec: float = 0.3,
        down_min_sec_cut: float = 0.6,          # when a frame border cuts the body off
        inactivity_sec: float = 4.0,
        recovery_sec: float = 1.5,
        horizontal_angle: float = 35.0,
        compact_aspect: float = 1.0,            # width / height of a body lying toward the camera
        sitting_torso_ratio: float = 0.22,      # torso length / upright height of someone sitting up
        upright_angle: float = 65.0,
        min_head_drop: float = 0.25,            # fractions of the reference height
        min_head_drop_compact: float = 0.55,    # sitting on the floor keeps the head ~0.5 up
        min_head_drop_occluded: float = 0.35,
        min_hip_drop: float = 0.2,
        hip_only_drop: float = 0.35,            # hip drop that counts when the head is not visible
        violent_collapse_rate: float = 150.0,   # deg/s; waives the hip check (see _is_down)
        occluded_confirm_sec: float = 0.7,
        baseline_window_sec: float = 1.2,
        ref_height_window_sec: float = 2.0
    ):
        self.descent_velocity_thresh = descent_velocity_thresh
        self.collapse_rate_thresh = collapse_rate_thresh
        self.collapse_min_velocity = collapse_min_velocity
        self.trigger_min_sec = trigger_min_sec
        self.confirm_window_sec = confirm_window_sec
        self.down_min_sec = down_min_sec
        self.down_min_sec_cut = down_min_sec_cut
        self.inactivity_sec = inactivity_sec
        self.recovery_sec = recovery_sec
        self.horizontal_angle = horizontal_angle
        self.compact_aspect = compact_aspect
        self.sitting_torso_ratio = sitting_torso_ratio
        self.upright_angle = upright_angle
        self.min_head_drop = min_head_drop
        self.min_head_drop_compact = min_head_drop_compact
        self.min_head_drop_occluded = min_head_drop_occluded
        self.min_hip_drop = min_hip_drop
        self.hip_only_drop = hip_only_drop
        self.violent_collapse_rate = violent_collapse_rate
        self.occluded_confirm_sec = occluded_confirm_sec
        self.baseline_window_sec = baseline_window_sec
        self.ref_height_window_sec = ref_height_window_sec

    def step(
        self,
        tracklet: Tracklet,
        features: Dict[str, Any],
        occlusion_stats: Dict[str, float],
        timestamp: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Advances the state machine for a track that was detected in this frame."""
        t = tracklet.time_history[-1] if timestamp is None else timestamp
        dt = self._advance_clock(tracklet, t)
        fsm = tracklet.fsm
        fsm["hidden_time"] = 0.0
        state = tracklet.state
        event_trigger = ""

        if state == self.STATE_MONITORING:
            self._update_ref_height(tracklet, t, features["bbox_height"])
            v_y = features["vertical_velocity"]
            v_thresh = self.descent_velocity_thresh * (0.8 if tracklet.in_occlusion_zone else 1.0)
            fast = tracklet.confirmed and (v_y > v_thresh or (
                features["angle_valid"]
                and features["angular_velocity"] > self.collapse_rate_thresh
                and v_y > self.collapse_min_velocity
            ))
            fsm["fast_time"] = fsm.get("fast_time", 0.0) + dt if fast else 0.0
            if fast and fsm["fast_time"] + 0.01 >= self.trigger_min_sec:
                self._begin_event(tracklet, t)
                state = self.STATE_RAPID_DESCENT
                event_trigger = f"Rapid descent (v={v_y:+.2f} h/s, angle {features['torso_angle']:.0f} deg)"

        elif state == self.STATE_RAPID_DESCENT:
            if features["angle_valid"]:
                fsm["peak_omega"] = max(fsm.get("peak_omega", 0.0), features["angular_velocity"])
            down = self._is_down(tracklet, features)
            if down:
                fsm["down_time"] += dt
                # A body cut off by a frame border (e.g. someone walking out past the bottom
                # edge) can briefly look low and wide, so it must stay visibly down for longer
                fsm["down_clear"] = not features["truncated"]
            elif down is not None:
                fsm["down_time"] = 0.0
                fsm["down_clear"] = False
            # (down is None: too little of the body visible to tell; keep the count as it is)
            needed = self.down_min_sec if fsm.get("down_clear", False) else self.down_min_sec_cut
            if down and fsm["down_time"] + 0.01 >= needed:
                state = self._enter_fall(tracklet)
                event_trigger = f"[!] FALL DETECTED (angle {features['torso_angle']:.0f} deg)"
            elif t - fsm["event_t"] > self.confirm_window_sec:
                state = self.STATE_MONITORING
                fsm["fast_time"] = 0.0
                event_trigger = "Descent ended upright (bend / sit) -> ignored"

        elif state in (self.STATE_FALL_DETECTED, self.STATE_IMPACT, self.STATE_CONFIRMED_FALL):
            tracklet.state_timer += dt
            upright = self._is_upright(tracklet, features)
            # Upright evidence accumulates; only a clear "down" observation resets it (a missed
            # detection or an ambiguous pose while getting up must not restart the count)
            if upright:
                fsm["recover_time"] = fsm.get("recover_time", 0.0) + dt
            elif self._is_down(tracklet, features):
                fsm["recover_time"] = 0.0
            if fsm.get("recover_time", 0.0) + 0.01 >= self.recovery_sec:
                state = self.STATE_MONITORING
                fsm["fast_time"] = 0.0
                event_trigger = "Subject back upright -> Reset"
            elif (state != self.STATE_CONFIRMED_FALL and not upright
                  and tracklet.state_timer >= self.inactivity_sec):
                # Never escalate while the person is in the middle of getting back up
                state = self.STATE_CONFIRMED_FALL
                event_trigger = "[EMERGENCY] CONFIRMED FALL - subject did not get up"

        if state == self.STATE_MONITORING:
            tracklet.seen_down = False
        else:
            down_now = self._is_down(tracklet, features)
            if down_now is not None:
                tracklet.seen_down = down_now
        tracklet.state = state
        return state, self._info(tracklet, event_trigger)

    def step_missing(self, tracklet: Tracklet, timestamp: float) -> Tuple[str, Dict[str, Any]]:
        """Advances timers for a track that was NOT detected in this frame (no new evidence)."""
        dt = self._advance_clock(tracklet, timestamp)
        fsm = tracklet.fsm
        state = tracklet.state
        event_trigger = ""

        if state == self.STATE_RAPID_DESCENT:
            if fsm.get("down_time", 0.0) > 0.0 and fsm.get("down_clear", False):
                # Last seen down on the floor, then lost: lying bodies are often missed by
                # the detector, so the last observed posture is assumed to persist
                fsm["down_time"] += dt
                if fsm["down_time"] + 0.01 >= self.down_min_sec:
                    state = self._enter_fall(tracklet)
                    event_trigger = "[!] FALL DETECTED (subject lost from view while down)"
            elif tracklet.in_occlusion_zone:
                # Dropped fast and vanished behind known furniture: treat as down
                fsm["hidden_time"] = fsm.get("hidden_time", 0.0) + dt
                if fsm["hidden_time"] >= self.occluded_confirm_sec:
                    state = self._enter_fall(tracklet)
                    event_trigger = f"[!] FALL DETECTED behind {tracklet.occlusion_zone_name}"
            elif timestamp - fsm["event_t"] > self.confirm_window_sec:
                state = self.STATE_MONITORING
                fsm["fast_time"] = 0.0
        elif state in (self.STATE_FALL_DETECTED, self.STATE_IMPACT, self.STATE_CONFIRMED_FALL):
            tracklet.state_timer += dt
            getting_up = fsm.get("recover_time", 0.0) > 0.0
            if (state != self.STATE_CONFIRMED_FALL and not getting_up
                    and tracklet.state_timer >= self.inactivity_sec):
                state = self.STATE_CONFIRMED_FALL
                event_trigger = "[EMERGENCY] CONFIRMED FALL - subject still hidden"

        tracklet.state = state
        return state, self._info(tracklet, event_trigger)

    # ------------------------------------------------------------------ helpers

    def _advance_clock(self, tracklet: Tracklet, t: float) -> float:
        last = tracklet.fsm.get("clock")
        tracklet.fsm["clock"] = t
        if last is None:
            return 0.0
        return float(min(max(t - last, 0.0), 0.5))

    def _update_ref_height(self, tracklet: Tracklet, t: float, height: float):
        log = tracklet.fsm.setdefault("heights", deque())
        log.append((t, height))
        while log and log[0][0] < t - self.ref_height_window_sec:
            log.popleft()
        tracklet.ref_height = max(h for _, h in log)

    def _begin_event(self, tracklet: Tracklet, t: float):
        """Stores the pre-descent posture that 'down' / 'upright' are measured against."""
        since = t - self.baseline_window_sec
        window = [i for i, ts in enumerate(tracklet.time_history) if ts >= since] or [len(tracklet.time_history) - 1]
        tops = [tracklet.bbox_history[i][1] for i in window]
        hips = [tracklet.hip_history[i] for i in window if tracklet.hip_history[i] is not None]
        uppers = [tracklet.upper_history[i] for i in window if tracklet.upper_history[i] is not None]
        tracklet.fsm.update({
            "event_t": t,
            "top0": min(tops),
            "top0_clipped": any(tracklet.top_clipped_history[i] for i in window),
            "hip0": min(hips) if hips else None,
            "upper0": min(uppers) if uppers else None,
            "h0": max(tracklet.ref_height, 1.0),
            "down_time": 0.0,
            "hidden_time": 0.0,
            "peak_omega": 0.0
        })

    def _enter_fall(self, tracklet: Tracklet) -> str:
        tracklet.state_timer = 0.0
        tracklet.fsm["recover_time"] = 0.0
        return self.STATE_FALL_DETECTED

    def _head_drop(self, tracklet: Tracklet, features: Dict[str, Any]) -> Optional[float]:
        """How far the top of the body dropped since the descent began, or None if unknown."""
        fsm = tracklet.fsm
        if not fsm.get("top0_clipped") and not features["truncated_top"]:
            return (features["top_y"] - fsm["top0"]) / fsm["h0"]
        # The head was outside the picture: use the visible head / shoulder joints instead
        if features["upper_y"] is not None and fsm.get("upper0") is not None:
            return (features["upper_y"] - fsm["upper0"]) / fsm["h0"]
        return None

    def _is_down(self, tracklet: Tracklet, features: Dict[str, Any]) -> Optional[bool]:
        """Is the body down on the floor? None when this frame shows too little to tell."""
        fsm = tracklet.fsm
        head_drop = self._head_drop(tracklet, features)
        hip_drop = None
        if features["hip_y"] is not None and fsm.get("hip0") is not None:
            hip_drop = (features["hip_y"] - fsm["hip0"]) / fsm["h0"]
        if tracklet.in_occlusion_zone:
            return None if head_drop is None else head_drop > self.min_head_drop_occluded

        horizontal = features["angle_valid"] and features["torso_angle"] < self.horizontal_angle
        # A long, upright torso is someone sitting (e.g. on the floor), whatever the box shape;
        # a body lying toward the camera shows a short (foreshortened) torso instead
        sitting_up = (features["angle_valid"] and features["head_up"] and features["torso_angle"] > self.upright_angle
                      and features["torso_ratio"] >= self.sitting_torso_ratio)
        compact = not sitting_up and (features["aspect_ratio"] > self.compact_aspect
                                      or features["bbox_height"] / fsm["h0"] < 0.55)

        if head_drop is None:
            # Head out of view (e.g. beyond the top of the picture): only a large hip drop tells
            if hip_drop is None:
                return None
            return hip_drop >= self.hip_only_drop and (horizontal or compact)

        if hip_drop is not None and hip_drop < self.min_hip_drop:
            # Hips barely lowered (bending over, lying on a bed, or a camera looking down
            # from the ceiling). Still a fall if the torso slammed flat and the body ended
            # wide: people do not lie down at that rotation speed.
            violent = fsm.get("peak_omega", 0.0) >= self.violent_collapse_rate
            flat_and_wide = (features["angle_valid"] and features["torso_angle"] < 30.0
                             and features["aspect_ratio"] > 1.3)
            if not (violent and flat_and_wide and head_drop > self.min_head_drop):
                return False
        return (horizontal and head_drop > self.min_head_drop) or (compact and head_drop > self.min_head_drop_compact)

    def _is_upright(self, tracklet: Tracklet, features: Dict[str, Any]) -> bool:
        head_drop = self._head_drop(tracklet, features)
        if head_drop is not None and head_drop < 0.2:
            return True
        return (
            features["angle_valid"]
            and features["head_up"]
            and features["torso_angle"] > self.upright_angle
            and (features["truncated"] or features["bbox_height"] / tracklet.fsm["h0"] > 0.65)
        )

    def _info(self, tracklet: Tracklet, event_trigger: str) -> Dict[str, Any]:
        countdown = 0.0
        if tracklet.state == self.STATE_FALL_DETECTED:
            countdown = max(0.0, self.inactivity_sec - tracklet.state_timer)
        return {
            "state_timer": tracklet.state_timer,
            "countdown": countdown,
            "event_trigger": event_trigger
        }


# ======================================================================================
# 6. TELEMETRY & HUD VISUALIZER
# ======================================================================================
class Visualizer:
    COLOR_NORMAL = (50, 220, 100)
    COLOR_OCCLUDED = (0, 165, 255)
    COLOR_DANGER = (0, 0, 255)
    COLOR_BG_PANEL = (20, 24, 30)
    COLOR_ZONE = (180, 100, 50)

    STATE_COLORS = {
        "MONITORING": (50, 220, 100),
        "RAPID_DESCENT": (0, 200, 255),
        "FALL_DETECTED": (0, 0, 255),
        "IMPACT": (0, 0, 255),
        "INACTIVITY_CHECK": (0, 140, 255),
        "CONFIRMED_FALL": (0, 0, 255)
    }

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render(
        self,
        frame: np.ndarray,
        tracklets: List[Tracklet],
        tracklet_data: Dict[int, Dict[str, Any]],
        occlusion_zones: Optional[List[Dict[str, Any]]] = None,
        fps_val: float = 0.0
    ) -> np.ndarray:
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]

        # 1. Render Occlusion Zones
        if occlusion_zones:
            overlay = vis_frame.copy()
            for zone in occlusion_zones:
                poly = zone["polygon"]
                cv2.fillPoly(overlay, [poly], self.COLOR_ZONE)
                cv2.polylines(vis_frame, [poly], isClosed=True, color=(255, 200, 100), thickness=2)
                moments = cv2.moments(poly)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    cv2.putText(vis_frame, f"ZONE: {zone['name']}", (cx - 40, cy),
                                self.font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.25, vis_frame, 0.75, 0, vis_frame)

        # 2. Render Top Status Bar
        visible = sum(1 for t in tracklets if not tracklet_data.get(t.track_id, {}).get("lost"))
        self._render_top_bar(vis_frame, fps_val, visible)

        # 3. Render Person Overlays
        for track in tracklets:
            t_id = track.track_id
            if t_id not in tracklet_data:
                continue

            data = tracklet_data[t_id]
            imputed_kps = data["imputed_kps"]
            occlusion_mask = data["occlusion_mask"]
            features = data["features"]
            state = data["state"]
            state_info = data["state_info"]
            occ_stats = data["occlusion_stats"]

            # Bounding box
            bbox = track.bbox_history[-1]

            if data.get("lost"):
                # Alarm still active but the person is not detected in this frame: mark where
                # they were last seen instead of freezing an old skeleton on screen
                self._draw_last_seen(vis_frame, bbox, t_id, state)
                self._draw_alert_banner(vis_frame, state, state_info, t_id)
                continue

            # Skeleton
            self._draw_skeleton(vis_frame, imputed_kps, occlusion_mask, state, bbox)

            # BBox & Card
            color = self.STATE_COLORS.get(state, self.COLOR_NORMAL)
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3 if state != "MONITORING" else 2)
            self._draw_person_card(vis_frame, x1, y1, t_id, state, features, occ_stats, state_info)

            # Alert Banner
            if state in ["FALL_DETECTED", "IMPACT", "INACTIVITY_CHECK", "CONFIRMED_FALL"]:
                self._draw_alert_banner(vis_frame, state, state_info, t_id)

        return vis_frame

    def _draw_last_seen(self, frame: np.ndarray, bbox: List[float], track_id: int, state: str):
        color = self.STATE_COLORS.get(state, self.COLOR_DANGER)
        x1, y1, x2, y2 = map(int, bbox)
        dash = 10
        for x in range(x1, x2, dash * 2):
            cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, 2)
            cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, 2)
        for y in range(y1, y2, dash * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, 2)
            cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, 2)
        cv2.putText(frame, f"PERSON #{track_id} last seen here (not visible)", (x1, max(15, y1 - 8)),
                    self.font, 0.45, color, 1, cv2.LINE_AA)

    def _draw_skeleton(self, frame: np.ndarray, kps: np.ndarray, occ_mask: np.ndarray, state: str, bbox: List[float]):
        state_color = self.STATE_COLORS.get(state, self.COLOR_NORMAL)
        bbox_w = max(bbox[2] - bbox[0], 20.0)
        bbox_h = max(bbox[3] - bbox[1], 20.0)
        max_allowed_bone_len = np.hypot(bbox_w, bbox_h) * 0.85

        def near_box(x: float, y: float) -> bool:
            return (bbox[0] - bbox_w * 0.3 <= x <= bbox[2] + bbox_w * 0.3
                    and bbox[1] - bbox_h * 0.3 <= y <= bbox[3] + bbox_h * 0.3)

        for p1, p2 in OcclusionEngine.SKELETON_PAIRS:
            x1, y1, _ = kps[p1]
            x2, y2, _ = kps[p2]
            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                continue
            if not (near_box(x1, y1) and near_box(x2, y2)):
                continue
            
            # Prevent stray lines across screen
            bone_dist = np.hypot(x1 - x2, y1 - y2)
            if bone_dist > max_allowed_bone_len:
                continue

            is_occ = occ_mask[p1] or occ_mask[p2]
            bone_color = self.COLOR_OCCLUDED if is_occ else state_color
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), bone_color, 2 if is_occ else 3, cv2.LINE_AA)

        for i in range(17):
            x, y, _ = kps[i]
            if x == 0 and y == 0:
                continue
            if not near_box(x, y):
                continue

            pt = (int(x), int(y))
            if occ_mask[i]:
                cv2.circle(frame, pt, 5, self.COLOR_OCCLUDED, 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, state_color, -1, cv2.LINE_AA)

    def _draw_person_card(self, frame: np.ndarray, x1: int, y1: int, track_id: int, state: str,
                          features: Dict[str, float], occ_stats: Dict[str, float], state_info: Dict[str, Any]):
        angle = features.get("torso_angle", 90.0)
        v_y = features.get("vertical_velocity", 0.0)
        card_w, card_h = 220, 85
        card_x = max(10, x1)
        card_y = max(45, y1 - card_h - 10)

        overlay = frame.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 30, 38), -1)
        state_color = self.STATE_COLORS.get(state, self.COLOR_NORMAL)
        cv2.rectangle(overlay, (card_x, card_y), (card_x + 5, card_y + card_h), state_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        display_state = state
        if state == "MONITORING" and angle < 45.0:
            display_state = "MONITORING (RESTING)"

        cv2.putText(frame, f"PERSON #{track_id} | {display_state}", (card_x + 12, card_y + 18),
                    self.font, 0.42, state_color, 1, cv2.LINE_AA)
        angle_text = f"Torso Angle: {angle:.1f} deg" if features.get("angle_valid", True) else "Torso Angle: -- (unclear)"
        cv2.putText(frame, angle_text, (card_x + 12, card_y + 36),
                    self.font, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        v_sign = "+" if v_y >= 0 else ""
        v_color = self.COLOR_DANGER if v_y > 1.0 else (200, 200, 200)
        cv2.putText(frame, f"V_descent: {v_sign}{v_y:.2f} h/s", (card_x + 12, card_y + 54),
                    self.font, 0.38, v_color, 1, cv2.LINE_AA)

        if occ_stats.get("is_partially_occluded", False):
            occ_pct = int((1.0 - occ_stats['total_visibility']) * 100)
            cv2.putText(frame, f"Occlusion: {occ_pct}% (IMPUTED)", (card_x + 12, card_y + 72),
                        self.font, 0.36, self.COLOR_OCCLUDED, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Visibility: 100% Clear", (card_x + 12, card_y + 72),
                        self.font, 0.36, (120, 200, 120), 1, cv2.LINE_AA)

    def _draw_alert_banner(self, frame: np.ndarray, state: str, state_info: Dict[str, Any], track_id: int):
        h, w = frame.shape[:2]
        banner_h = 75
        banner_y = h - banner_h - 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, banner_y), (w - 20, banner_y + banner_h), (15, 15, 220), -1)
        cv2.rectangle(overlay, (20, banner_y), (w - 20, banner_y + banner_h), (255, 255, 255), 2)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        if state in ["FALL_DETECTED", "IMPACT", "INACTIVITY_CHECK"]:
            countdown = state_info.get("countdown", 1.5)
            text = f"[ALERT] FALL DETECTED (PERSON #{track_id})"
            sub = f"Escalates to emergency in {countdown:.1f}s unless the person gets back up."
            cv2.putText(frame, text, (35, banner_y + 30), self.font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, sub, (35, banner_y + 55), self.font, 0.45, (200, 230, 255), 1, cv2.LINE_AA)
        else:
            text = f"[EMERGENCY] CONFIRMED FALL DETECTED (PERSON #{track_id})"
            sub = "Alert dispatched to caregiver / emergency contacts"
            cv2.putText(frame, text, (35, banner_y + 30), self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, sub, (35, banner_y + 55), self.font, 0.45, (220, 240, 255), 1, cv2.LINE_AA)

    def _render_top_bar(self, frame: np.ndarray, fps: float, count: int):
        w = frame.shape[1]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), self.COLOR_BG_PANEL, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "FALL DETECTION", (15, 21),
                    self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        status_str = f"FPS: {fps:.1f}  |  Tracking: {count} Subject(s)  |  Edge CPU Mode"
        cv2.putText(frame, status_str, (w - 380, 21), self.font, 0.45, (160, 220, 160), 1, cv2.LINE_AA)


# ======================================================================================
# 7. SYNTHETIC OCCLUSION TEST VIDEO GENERATOR
# ======================================================================================
def generate_fall_test_video(output_path: str = "demo_fall_with_occlusion.mp4", duration_sec: int = 14, fps: int = 30):
    width, height = 854, 480
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    bg_color = (240, 240, 245)
    floor_color = (210, 190, 170)
    table_color = (60, 90, 140)
    table_x1, table_y1 = 300, 310
    table_x2, table_y2 = 620, 440

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Floor
        cv2.rectangle(frame, (0, 360), (width, height), floor_color, -1)
        cv2.line(frame, (0, 360), (width, 360), (180, 160, 140), 2)

        if t < 5.0:
            px = int(120 + t * 65)
            head_y = 160
            torso_angle = 90.0
            leg_swing = np.sin(t * 8) * 20
        elif t < 6.0:
            fall_prog = (t - 5.0) / 1.0
            px = int(445 + fall_prog * 30)
            head_y = int(160 + fall_prog * 180)
            torso_angle = 90.0 - fall_prog * 75.0
            leg_swing = 0
        else:
            px = 475
            head_y = 345
            torso_angle = 12.0
            leg_swing = 0

        skin_color = (180, 190, 220)
        cloth_color = (120, 60, 40)
        pants_color = (50, 50, 80)
        rad = np.radians(torso_angle)
        torso_len = 90

        hx, hy = px, head_y
        hip_x = int(hx - torso_len * np.cos(rad))
        hip_y = int(hy + torso_len * np.sin(rad))
        knee_x = int(hip_x - 45 * np.cos(rad) + leg_swing * 0.3)
        knee_y = int(hip_y + 45 * np.sin(rad))
        foot_x = int(knee_x - 40 * np.cos(rad) - leg_swing * 0.5)
        foot_y = min(height - 10, int(knee_y + 40 * np.sin(rad)))

        cv2.line(frame, (hip_x, hip_y), (knee_x, knee_y), pants_color, 14)
        cv2.line(frame, (knee_x, knee_y), (foot_x, foot_y), pants_color, 12)
        cv2.line(frame, (hx, hy), (hip_x, hip_y), cloth_color, 18)
        cv2.circle(frame, (hx, hy), 16, skin_color, -1)
        cv2.line(frame, (hx, hy + 20), (int(hx + 25 * np.cos(rad)), int(hy + 20 + 35 * np.sin(rad))), cloth_color, 8)

        # Draw Table Barrier over subject's lower body
        cv2.rectangle(frame, (table_x1, table_y1), (table_x2, table_y2), table_color, -1)
        cv2.rectangle(frame, (table_x1, table_y1), (table_x2, table_y2), (40, 60, 100), 3)
        cv2.putText(frame, "[OCCLUSION BARRIER: TABLE]", (table_x1 + 35, table_y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        status_text = "Phase: Walking" if t < 5.0 else ("Phase: SLIPPING / FALLING" if t < 6.0 else "Phase: POST-FALL IMMOBILITY")
        cv2.putText(frame, f"TEST SCENARIO: {status_text}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"[TestGenerator] Synthetic fall test video generated at: {output_path}")


# ======================================================================================
# 8. MAIN ORCHESTRATOR & PIPELINE RUNNER
# ======================================================================================
class FallDetectionPipeline:
    def __init__(self, model_name: str = "yolov8n-pose.pt", conf_thresh: float = 0.35, device: str = None, fps: float = 30.0):
        self.fps = fps              # Shown in the HUD; runners overwrite it with the processing FPS
        self.video_fps = fps        # Used to derive timestamps when none are supplied
        self._frame_index = 0
        self.detector = PoseDetector(model_name=model_name, conf_thresh=conf_thresh, device=device)
        self.tracker = SimpleTracker()
        self.occlusion_engine = OcclusionEngine(conf_threshold=0.25)
        self.kinematics = KinematicExtractor()
        self.state_machine = FallStateMachine(inactivity_sec=4.0)
        self.visualizer = Visualizer()

    def add_furniture_zone(self, name: str, polygon: List[Tuple[int, int]]):
        self.occlusion_engine.add_occlusion_zone(name, polygon)

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        timestamp: capture time in seconds (live sources) or position in the video (files).
        Defaults to frame_index / fps, which is only correct when no frames are skipped.
        """
        if timestamp is None:
            timestamp = self._frame_index / max(self.video_fps, 1.0)
        self._frame_index += 1

        frame_size = (frame.shape[1], frame.shape[0])
        self.kinematics.frame_size = frame_size
        detections = self.detector.detect(frame)
        tracklets = self.tracker.update(detections, timestamp, frame_size)

        drawn: List[Tracklet] = []
        tracklet_data: Dict[int, Dict[str, Any]] = {}
        alerts: List[Dict[str, Any]] = []

        for track in tracklets:
            t_id = track.track_id

            if track.is_visible:
                curr_bbox = track.bbox_history[-1]
                raw_kps = track.raw_kps_history[-1]

                imputed_kps, occ_mask, occ_stats = self.occlusion_engine.process_keypoints(
                    current_kps=raw_kps,
                    history_kps=track.imputed_kps_history[:-1],
                    bbox=curr_bbox
                )
                track.imputed_kps_history[-1] = imputed_kps

                centroid = ((curr_bbox[0] + curr_bbox[2]) / 2.0, (curr_bbox[1] + curr_bbox[3]) / 2.0)
                in_zone, zone_name = self.occlusion_engine.is_in_occlusion_zone(centroid)
                track.in_occlusion_zone = in_zone
                track.occlusion_zone_name = zone_name

                features = self.kinematics.extract_features(track)
                state, state_info = self.state_machine.step(track, features, occ_stats, timestamp)

                data = {
                    "imputed_kps": imputed_kps,
                    "occlusion_mask": occ_mask,
                    "occlusion_stats": occ_stats,
                    "features": features,
                    "state": state,
                    "state_info": state_info
                }
                track.last_data = data
                if not track.confirmed:
                    continue  # tentative (possible ghost): never drawn, never alarms
            else:
                # Not detected this frame: no new evidence, only timers advance. A lost person
                # is only shown while an alarm is active, as a "last seen" outline (no skeleton).
                if track.last_data is None or not track.confirmed:
                    continue
                state, state_info = self.state_machine.step_missing(track, timestamp)
                if state not in ALERT_STATES:
                    continue
                data = dict(track.last_data, state=state, state_info=state_info, lost=True)

            drawn.append(track)
            tracklet_data[t_id] = data
            if state in ALERT_STATES:
                alerts.append({"track_id": t_id, "state": state, "info": state_info, "features": data["features"]})

        rendered_frame = self.visualizer.render(
            frame=frame,
            tracklets=drawn,
            tracklet_data=tracklet_data,
            occlusion_zones=self.occlusion_engine.occlusion_zones,
            fps_val=self.fps
        )
        return rendered_frame, alerts


def run_pipeline(
    source: str = "demo",
    save_path: str = "fall_detection_output.mp4",
    show: bool = True,
    conf: float = 0.35,
    device: str = None
):
    print("=" * 75)
    print("  AI PERSON & ELDERLY FALL DETECTION SYSTEM (ALL-IN-ONE)")
    print("  With Advanced Occlusion Resilience & Kinematic State Machine")
    print("=" * 75)

    if source == "demo":
        demo_file = "demo_fall_with_occlusion.mp4"
        if not os.path.exists(demo_file):
            print("[Pipeline] Generating synthetic test video with occlusion barrier...")
            generate_fall_test_video(demo_file)
        source = demo_file

    is_cam = str(source).isdigit()
    if is_cam:
        # On Windows, DirectShow backend (CAP_DSHOW) is fast and reliable
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        print(f"[Error] Could not open video source: '{source}'. Check file path or camera permissions.")
        return

    # Read first frame to ensure camera/video is actively streaming valid dimensions
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        print(f"[Error] Could not read frames from video source: '{source}'")
        cap.release()
        return

    height, width = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 120:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0

    print(f"[Pipeline] Initialized Source: {source} ({width}x{height} @ {fps:.1f} FPS)")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"[Pipeline] Output will be saved to: {save_path}")

    pipeline = FallDetectionPipeline(conf_thresh=conf, device=device, fps=fps)

    if "demo" in str(source):
        pipeline.add_furniture_zone("Coffee Table", [(300, 310), (620, 310), (620, 440), (300, 440)])

    frame_count = 0
    t_start = time.time()
    window_name = "AI Fall Detection & Occlusion Resilience"

    if show:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # Display scaled window if large, keeping aspect ratio
            display_w = min(width, 1024)
            display_h = int(height * (display_w / max(width, 1)))
            cv2.resizeWindow(window_name, display_w, display_h)
        except Exception:
            pass

    last_alert_state: Dict[int, str] = {}

    try:
        frame = first_frame
        while True:
            if frame is None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

            frame_count += 1
            t_frame_start = time.time()

            # Real capture time for cameras; position in the file for videos
            timestamp = time.perf_counter() if is_cam else (frame_count - 1) / fps
            vis_frame, alerts = pipeline.process_frame(frame, timestamp)

            t_elapsed = time.time() - t_frame_start
            current_fps = 1.0 / max(t_elapsed, 1e-4)
            pipeline.fps = current_fps

            # Report state changes only, not every frame an alarm stays active
            active = {alert["track_id"]: alert for alert in alerts}
            for track_id, alert in active.items():
                if last_alert_state.get(track_id) != alert["state"]:
                    print(f"🚨 [EMERGENCY ALERT] Frame {frame_count}: {alert['state']} for Person #{track_id}! "
                          f"(Torso Angle: {alert['features']['torso_angle']:.1f}°)")
            last_alert_state = {track_id: alert["state"] for track_id, alert in active.items()}

            if writer:
                writer.write(vis_frame)

            if show:
                cv2.imshow(window_name, vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    print("[Pipeline] User requested exit.")
                    break

            if frame_count % 30 == 0 and total_frames > 0:
                print(f"[Pipeline] Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

            # Reset frame to read next frame in next iteration
            frame = None

    finally:
        cap.release()
        if writer:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    total_time = time.time() - t_start
    avg_fps = frame_count / max(total_time, 1e-4)
    print("=" * 75)
    print(f"  Execution Complete! Processed {frame_count} frames in {total_time:.2f}s (Avg FPS: {avg_fps:.1f})")
    if save_path:
        print(f"  Processed video saved at: {os.path.abspath(save_path)}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Fall Detection & Occlusion System (All-in-One)")
    parser.add_argument("--source", type=str, default="demo", help="Path to video file, webcam index (0), or 'demo'")
    parser.add_argument("--save", type=str, default="fall_detection_output.mp4", help="Path to save output video")
    parser.add_argument("--show", action="store_true", default=False, help="Display live OpenCV window")
    parser.add_argument("--conf", type=float, default=0.35, help="Pose detection confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="'cpu', 'cuda', or None for auto")

    args = parser.parse_args()
    run_pipeline(source=args.source, save_path=args.save, show=args.show, conf=args.conf, device=args.device)
