"""
Thread bridge between the worker-thread `SessionListener` callbacks and the
Tkinter main loop (Tk widgets must only be touched from the UI thread).
"""

import queue
import threading
from typing import List, Optional, Tuple, Any

from app.session import SessionListener, FrameUpdate, AlertEvent, SessionSummary


class QueuedListener(SessionListener):
    def __init__(self):
        self._events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[FrameUpdate] = None

    # Worker-thread side -------------------------------------------------------

    def on_status(self, message: str) -> None:
        self._events.put(("status", message))

    def on_frame(self, update: FrameUpdate) -> None:
        # Keep only the newest frame; the UI renders at its own pace
        with self._frame_lock:
            self._latest_frame = update

    def on_alert(self, event: AlertEvent) -> None:
        self._events.put(("alert", event))

    def on_finished(self, summary: SessionSummary) -> None:
        self._events.put(("finished", summary))

    # UI-thread side -----------------------------------------------------------

    def take_frame(self) -> Optional[FrameUpdate]:
        with self._frame_lock:
            update, self._latest_frame = self._latest_frame, None
        return update

    def drain_events(self) -> List[Tuple[str, Any]]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events
