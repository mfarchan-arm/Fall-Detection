"""
Application logic layer (GUI-agnostic).

Everything in this package is free of any GUI toolkit imports so it can be driven
by the Tkinter front-end in `gui/`, a CLI, or tests.
"""
from app.services import (
    SourceKind,
    VideoEntry,
    POSE_MODELS,
    DEFAULT_POSE_MODEL,
    SessionConfig,
    discover_videos,
    validate_video_file,
    parse_fps_override,
    open_with_system,
    build_webcam_config,
    build_video_config,
    build_demo_config,
)
from app.session import (
    DetectionSession,
    SessionListener,
    FrameUpdate,
    AlertEvent,
    SessionSummary,
)

__all__ = [
    "SourceKind",
    "VideoEntry",
    "POSE_MODELS",
    "DEFAULT_POSE_MODEL",
    "SessionConfig",
    "discover_videos",
    "validate_video_file",
    "parse_fps_override",
    "open_with_system",
    "build_webcam_config",
    "build_video_config",
    "build_demo_config",
    "DetectionSession",
    "SessionListener",
    "FrameUpdate",
    "AlertEvent",
    "SessionSummary",
]
