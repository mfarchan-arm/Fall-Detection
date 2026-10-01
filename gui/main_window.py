"""
Main Tkinter window for the Fall Detection System.

Presentation only: source selection, settings, embedded video preview, progress
and alert log. All detection work is delegated to `app.DetectionSession`.
"""

import os
import sys
import time
import ctypes
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List

from PIL import Image, ImageTk

from app import (
    SourceKind,
    POSE_MODELS,
    DEFAULT_POSE_MODEL,
    VideoEntry,
    SessionConfig,
    DetectionSession,
    AlertEvent,
    SessionSummary,
    FrameUpdate,
    discover_videos,
    validate_video_file,
    parse_fps_override,
    open_with_system,
    build_webcam_config,
    build_video_config,
    build_demo_config,
)
from gui.bridge import QueuedListener

APP_TITLE = "AI Person & Elderly Fall Detection System"
POLL_MS = 30
BANNER_HOLD_SEC = 6.0
FPS_AUTO = "Auto (from file)"

COLORS = {
    "bg": "#1e1f24",
    "panel": "#26282f",
    "fg": "#e6e6e6",
    "muted": "#9aa0a6",
    "accent": "#3b82f6",
    "danger": "#dc2626",
    "ok": "#16a34a",
    "video_bg": "#101114",
}

STATE_LABELS = {
    "FALL_DETECTED": "FALL DETECTED",
    "IMPACT": "IMPACT",
    "CONFIRMED_FALL": "CONFIRMED FALL",
}


class MainWindow:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.session: Optional[DetectionSession] = None
        self.listener: Optional[QueuedListener] = None
        self.videos: List[VideoEntry] = []
        self.last_output: Optional[str] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._banner_until = 0.0
        # Fonts are in points and scale on their own; pixel sizes go through px()
        self.ui_scale = max(root.winfo_fpixels("1i") / 96.0, 1.0)

        self.source_var = tk.StringVar(value=SourceKind.WEBCAM.value)
        self.cam_index_var = tk.IntVar(value=0)
        self.video_path_var = tk.StringVar()
        self.video_fps_var = tk.StringVar(value=FPS_AUTO)
        self.conf_var = tk.DoubleVar(value=0.35)
        self.save_var = tk.BooleanVar(value=True)
        self.model_labels = {f"{name}  ({desc})": name for name, desc in POSE_MODELS}
        self.model_var = tk.StringVar(value=next(k for k, v in self.model_labels.items() if v == DEFAULT_POSE_MODEL))
        self.status_var = tk.StringVar(value="Ready. Select an input source and press Start.")
        self.stats_var = tk.StringVar(value="")

        self._setup_window()
        self._setup_style()
        self._build_layout()
        self._on_source_changed()
        self.refresh_videos()
        self._set_running(False)

        self.root.after(POLL_MS, self._poll)

    # ------------------------------------------------------------------ setup

    def px(self, value: float) -> int:
        return int(round(value * self.ui_scale))

    def _setup_window(self) -> None:
        self.root.title(APP_TITLE)
        # Fit the scaled window inside the screen (leave room for the taskbar)
        width = min(self.px(1280), self.root.winfo_screenwidth() - self.px(40))
        height = min(self.px(780), self.root.winfo_screenheight() - self.px(80))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(self.px(980), width), min(self.px(620), height))
        self.root.configure(bg=COLORS["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.root.bind("<Escape>", lambda _e: self.stop())
        self.root.bind("<KeyPress-q>", self._on_q_pressed)

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        c = COLORS
        style.configure(".", background=c["bg"], foreground=c["fg"], fieldbackground=c["panel"])
        style.configure("TFrame", background=c["bg"])
        style.configure("Panel.TFrame", background=c["panel"])
        style.configure("TLabel", background=c["bg"], foreground=c["fg"])
        style.configure("Panel.TLabel", background=c["panel"], foreground=c["fg"])
        style.configure("Muted.TLabel", background=c["panel"], foreground=c["muted"])
        style.configure("Title.TLabel", background=c["bg"], font=("Segoe UI", 15, "bold"))
        style.configure("Subtitle.TLabel", background=c["bg"], foreground=c["muted"], font=("Segoe UI", 9))
        style.configure("TLabelframe", background=c["panel"], bordercolor="#3a3d46")
        style.configure("TLabelframe.Label", background=c["panel"], foreground=c["fg"], font=("Segoe UI", 10, "bold"))
        style.configure("TRadiobutton", background=c["panel"], foreground=c["fg"])
        style.map("TRadiobutton", background=[("active", c["panel"])])
        style.configure("TCheckbutton", background=c["panel"], foreground=c["fg"])
        style.map("TCheckbutton", background=[("active", c["panel"])])
        style.configure("TCombobox", fieldbackground=c["bg"], background="#32353d", foreground=c["fg"],
                        arrowcolor=c["fg"], selectbackground=c["bg"], selectforeground=c["fg"])
        style.map("TCombobox", fieldbackground=[("readonly", c["bg"])], foreground=[("readonly", c["fg"])],
                  selectbackground=[("readonly", c["bg"])], selectforeground=[("readonly", c["fg"])])
        # Drop-down list of the comboboxes
        self.root.option_add("*TCombobox*Listbox.background", c["bg"])
        self.root.option_add("*TCombobox*Listbox.foreground", c["fg"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", c["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "white")
        style.configure("TScale", background=c["panel"])
        style.configure("TButton", padding=6, background="#32353d", foreground=c["fg"], bordercolor="#3a3d46")
        style.map("TButton", background=[("disabled", c["panel"]), ("active", "#3f434d")],
                  foreground=[("disabled", c["muted"])])
        style.configure("Accent.TButton", background=c["accent"], foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("disabled", "#3a3d46"), ("active", "#2563eb")])
        style.configure("Danger.TButton", background=c["danger"], foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Danger.TButton", background=[("disabled", "#3a3d46"), ("active", "#b91c1c")])
        style.configure("Treeview", background=c["panel"], fieldbackground=c["panel"], foreground=c["fg"], rowheight=self.px(22))
        style.configure("Treeview.Heading", background="#32353d", foreground=c["fg"])
        style.configure("TProgressbar", troughcolor=c["panel"], background=c["accent"])

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, padding=(16, 12, 16, 4))
        header.pack(fill="x")
        ttk.Label(header, text=f"\U0001F6E1  {APP_TITLE}", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Advanced Occlusion Resilience & Kinematic Tracking",
                  style="Subtitle.TLabel").pack(anchor="w")

        body = ttk.Frame(self.root, padding=(16, 8, 16, 16))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_sidebar(body).grid(row=0, column=0, sticky="ns", padx=(0, 12))
        self._build_main_area(body).grid(row=0, column=1, sticky="nsew")

    def _build_sidebar(self, parent: tk.Widget) -> ttk.Frame:
        side = ttk.Frame(parent, width=self.px(330))

        # Input source ---------------------------------------------------------
        src = ttk.LabelFrame(side, text="Input Source", padding=10)
        src.pack(fill="x")

        for kind, text in (
            (SourceKind.WEBCAM, "\U0001F4F9  Live Webcam"),
            (SourceKind.VIDEO, "\U0001F4C1  Custom Video File"),
            (SourceKind.DEMO, "\U0001F9EA  Synthetic Occlusion Fall Demo"),
        ):
            ttk.Radiobutton(src, text=text, value=kind.value, variable=self.source_var,
                            command=self._on_source_changed).pack(anchor="w", pady=2)

        self.options_holder = ttk.Frame(src, style="Panel.TFrame")
        self.options_holder.pack(fill="x", pady=(8, 0))

        self.webcam_opts = ttk.Frame(self.options_holder, style="Panel.TFrame")
        ttk.Label(self.webcam_opts, text="Camera index:", style="Panel.TLabel").pack(side="left")
        ttk.Spinbox(self.webcam_opts, from_=0, to=9, width=5,
                    textvariable=self.cam_index_var).pack(side="left", padx=6)

        self.video_opts = ttk.Frame(self.options_holder, style="Panel.TFrame")
        ttk.Label(self.video_opts, text="Detected videos:", style="Muted.TLabel").pack(anchor="w")
        list_frame = ttk.Frame(self.video_opts, style="Panel.TFrame")
        list_frame.pack(fill="x", pady=4)
        self.video_list = tk.Listbox(
            list_frame, height=6, activestyle="none", exportselection=False,
            bg=COLORS["bg"], fg=COLORS["fg"], selectbackground=COLORS["accent"],
            highlightthickness=0, borderwidth=0, font=("Segoe UI", 9),
        )
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.video_list.yview)
        self.video_list.configure(yscrollcommand=scroll.set)
        self.video_list.pack(side="left", fill="x", expand=True)
        scroll.pack(side="right", fill="y")
        self.video_list.bind("<<ListboxSelect>>", self._on_video_selected)
        self.video_list.bind("<Double-Button-1>", lambda _e: self.start())

        btns = ttk.Frame(self.video_opts, style="Panel.TFrame")
        btns.pack(fill="x")
        ttk.Button(btns, text="Browse...", command=self.browse_video).pack(side="left")
        ttk.Button(btns, text="Refresh", command=self.refresh_videos).pack(side="left", padx=6)
        ttk.Label(self.video_opts, text="Selected file:", style="Muted.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Entry(self.video_opts, textvariable=self.video_path_var).pack(fill="x")
        fps_row = ttk.Frame(self.video_opts, style="Panel.TFrame")
        fps_row.pack(fill="x", pady=(8, 0))
        ttk.Label(fps_row, text="Frame rate:", style="Panel.TLabel").pack(side="left")
        ttk.Combobox(fps_row, textvariable=self.video_fps_var, width=16,
                     values=[FPS_AUTO, "15", "25", "30", "60"]).pack(side="left", padx=6)
        ttk.Label(
            self.video_opts, style="Muted.TLabel", wraplength=self.px(280), justify="left",
            text="Set the real recording rate if the file header is wrong "
                 "(e.g. Multiple Cameras Fall Dataset clips: 30).",
        ).pack(anchor="w", pady=(2, 0))

        self.demo_opts = ttk.Frame(self.options_holder, style="Panel.TFrame")
        ttk.Label(
            self.demo_opts, style="Muted.TLabel", wraplength=self.px(280), justify="left",
            text="Generates (if needed) and analyses a synthetic scene of a person "
                 "walking and falling behind a table occlusion barrier.",
        ).pack(anchor="w")

        # Settings -------------------------------------------------------------
        settings = ttk.LabelFrame(side, text="Settings", padding=10)
        settings.pack(fill="x", pady=(12, 0))

        ttk.Label(settings, text="Pose model", style="Panel.TLabel").pack(anchor="w")
        ttk.Combobox(settings, textvariable=self.model_var, values=list(self.model_labels),
                     state="readonly").pack(fill="x", pady=(2, 8))

        conf_row = ttk.Frame(settings, style="Panel.TFrame")
        conf_row.pack(fill="x")
        ttk.Label(conf_row, text="Detection confidence", style="Panel.TLabel").pack(side="left")
        self.conf_label = ttk.Label(conf_row, text="0.35", style="Panel.TLabel")
        self.conf_label.pack(side="right")
        ttk.Scale(settings, from_=0.10, to=0.90, variable=self.conf_var,
                  command=lambda v: self.conf_label.configure(text=f"{float(v):.2f}")).pack(fill="x", pady=4)
        ttk.Checkbutton(settings, text="Save annotated output video",
                        variable=self.save_var).pack(anchor="w", pady=(4, 0))

        # Actions --------------------------------------------------------------
        actions = ttk.Frame(side)
        actions.pack(fill="x", pady=(16, 0))
        self.start_btn = ttk.Button(actions, text="▶  Start Detection", style="Accent.TButton", command=self.start)
        self.start_btn.pack(fill="x")
        self.stop_btn = ttk.Button(actions, text="■  Stop", style="Danger.TButton", command=self.stop)
        self.stop_btn.pack(fill="x", pady=(6, 0))
        bottom_row = ttk.Frame(actions)
        bottom_row.pack(fill="x", pady=(6, 0))
        bottom_row.columnconfigure(0, weight=1)
        bottom_row.columnconfigure(1, weight=1)
        self.open_btn = ttk.Button(bottom_row, text="Open Output Video", command=self.open_output)
        self.open_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(bottom_row, text="Exit", command=self.on_exit).grid(row=0, column=1, sticky="ew", padx=(3, 0))

        ttk.Label(side, text="Tip: press Esc or Q to stop a running analysis.",
                  style="Subtitle.TLabel").pack(anchor="w", pady=(12, 0))
        return side

    def _build_main_area(self, parent: tk.Widget) -> ttk.Frame:
        main = ttk.Frame(parent)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        self.banner = tk.Label(main, text="", bg=COLORS["danger"], fg="white",
                               font=("Segoe UI", 13, "bold"), pady=6)

        self.video_label = tk.Label(
            main, bg=COLORS["video_bg"], fg=COLORS["muted"], font=("Segoe UI", 11),
            text="No video running.\nChoose a source on the left and press Start.",
        )
        self.video_label.grid(row=1, column=0, sticky="nsew")

        status = ttk.Frame(main, padding=(0, 8, 0, 0))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(status, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(status, textvariable=self.status_var).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(status, textvariable=self.stats_var, style="Subtitle.TLabel").grid(row=1, column=1, sticky="e")

        log = ttk.LabelFrame(main, text="Emergency Alert Log", padding=6)
        log.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        cols = ("time", "frame", "person", "state", "angle")
        self.alert_tree = ttk.Treeview(log, columns=cols, show="headings", height=5)
        for col, heading, width in (
            ("time", "Time", 90), ("frame", "Frame", 70), ("person", "Person", 70),
            ("state", "State", 160), ("angle", "Torso Angle", 100),
        ):
            self.alert_tree.heading(col, text=heading)
            self.alert_tree.column(col, width=self.px(width), anchor="center")
        self.alert_tree.tag_configure("alert", foreground="#f87171")
        self.alert_tree.pack(fill="x")
        return main

    # ---------------------------------------------------------------- actions

    def refresh_videos(self) -> None:
        self.videos = discover_videos()
        self.video_list.delete(0, "end")
        for entry in self.videos:
            marker = "★" if entry.is_benchmark else "•"  # star = sample_videos/
            self.video_list.insert("end", f"{marker} {os.path.basename(entry.path)}  ({entry.size_kb} KB)")
        if not self.videos:
            self.video_list.insert("end", "(no videos found — use Browse...)")

    def browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a video to analyse",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")],
        )
        if path:
            self.video_path_var.set(path)
            self.video_list.selection_clear(0, "end")

    def start(self) -> None:
        if self.session and self.session.is_running:
            return
        config = self._build_config()
        if config is None:
            return

        self.alert_tree.delete(*self.alert_tree.get_children())
        self._hide_banner()
        self.progress.configure(value=0, mode="indeterminate" if config.is_webcam else "determinate")
        if config.is_webcam:
            self.progress.start(15)
        self.stats_var.set("")
        self.last_output = None

        self.listener = QueuedListener()
        self.session = DetectionSession(config, self.listener)
        self.session.start()
        self._set_running(True)

    def stop(self) -> None:
        if self.session and self.session.is_running:
            self.status_var.set("Stopping...")
            self.session.stop()

    def open_output(self) -> None:
        if self.last_output and os.path.exists(self.last_output):
            open_with_system(self.last_output)

    def on_exit(self) -> None:
        if self.session and self.session.is_running:
            if not messagebox.askyesno(APP_TITLE, "Detection is still running. Stop it and exit?"):
                return
            self.session.stop()
            self.session.join(timeout=3.0)
        self.root.destroy()

    # ---------------------------------------------------------------- helpers

    def _build_config(self) -> Optional[SessionConfig]:
        kind = SourceKind(self.source_var.get())
        conf = round(float(self.conf_var.get()), 2)
        save = bool(self.save_var.get())
        model = self.model_labels[self.model_var.get()]

        if kind == SourceKind.WEBCAM:
            try:
                cam = int(self.cam_index_var.get())
            except (tk.TclError, ValueError):
                messagebox.showerror(APP_TITLE, "Camera index must be a number.")
                return None
            return build_webcam_config(cam, conf=conf, save=save, model_name=model)

        if kind == SourceKind.VIDEO:
            path = self.video_path_var.get()
            error = validate_video_file(path)
            if error:
                messagebox.showerror(APP_TITLE, error)
                return None
            try:
                fps_override = parse_fps_override(self.video_fps_var.get())
            except ValueError:
                messagebox.showerror(APP_TITLE, "Frame rate must be 'Auto' or a number between 1 and 240.")
                return None
            return build_video_config(path, conf=conf, save=save, model_name=model, fps_override=fps_override)

        return build_demo_config(conf=conf, save=save, model_name=model)

    def _on_q_pressed(self, event: tk.Event) -> None:
        # Don't hijack the key while the user is typing a path / camera index
        if not isinstance(event.widget, (tk.Entry, ttk.Entry)):
            self.stop()

    def _on_source_changed(self) -> None:
        for frame in (self.webcam_opts, self.video_opts, self.demo_opts):
            frame.pack_forget()
        {
            SourceKind.WEBCAM.value: self.webcam_opts,
            SourceKind.VIDEO.value: self.video_opts,
            SourceKind.DEMO.value: self.demo_opts,
        }[self.source_var.get()].pack(fill="x")

    def _on_video_selected(self, _event=None) -> None:
        sel = self.video_list.curselection()
        if sel and sel[0] < len(self.videos):
            self.video_path_var.set(self.videos[sel[0]].path)

    def _set_running(self, running: bool) -> None:
        self.start_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        self.open_btn.configure(state="normal" if (not running and self.last_output) else "disabled")

    def _show_banner(self, text: str) -> None:
        self.banner.configure(text=text)
        self.banner.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self._banner_until = time.time() + BANNER_HOLD_SEC

    def _hide_banner(self) -> None:
        self.banner.grid_remove()
        self._banner_until = 0.0

    # ------------------------------------------------------------ event loop

    def _poll(self) -> None:
        if self.listener is not None:
            for kind, payload in self.listener.drain_events():
                if kind == "status":
                    self.status_var.set(payload)
                elif kind == "alert":
                    self._handle_alert(payload)
                elif kind == "finished":
                    self._handle_finished(payload)

            update = self.listener.take_frame()
            if update is not None:
                self._render_frame(update)

        if self._banner_until and time.time() > self._banner_until:
            self._hide_banner()

        self.root.after(POLL_MS, self._poll)

    def _render_frame(self, update: FrameUpdate) -> None:
        box_w = max(self.video_label.winfo_width(), 1)
        box_h = max(self.video_label.winfo_height(), 1)
        h, w = update.frame.shape[:2]
        scale = min(box_w / w, box_h / h)
        size = (max(int(w * scale), 1), max(int(h * scale), 1))

        # LANCZOS keeps detail when shrinking; BICUBIC is smoother when enlarging low-res sources
        resample = Image.LANCZOS if scale < 1.0 else Image.BICUBIC
        image = Image.fromarray(update.frame[:, :, ::-1]).resize(size, resample)  # BGR -> RGB
        self._photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self._photo, text="")

        if update.total_frames > 0:
            self.progress.configure(value=update.frame_index / update.total_frames * 100)
            self.stats_var.set(f"Frame {update.frame_index}/{update.total_frames}  ·  {update.fps:.1f} FPS")
        else:
            self.stats_var.set(f"Frame {update.frame_index}  ·  {update.fps:.1f} FPS")

    def _handle_alert(self, event: AlertEvent) -> None:
        label = STATE_LABELS.get(event.state, event.state)
        self.alert_tree.insert(
            "", 0, tags=("alert",),
            values=(time.strftime("%H:%M:%S", time.localtime(event.timestamp)), event.frame_index,
                    f"#{event.track_id}", label, f"{event.torso_angle:.1f}°"),
        )
        self._show_banner(f"\U0001F6A8  {label} — Person #{event.track_id}  (frame {event.frame_index})")
        self.root.bell()

    def _handle_finished(self, summary: SessionSummary) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate", value=0 if summary.error else 100)
        self.last_output = summary.save_path
        self._set_running(False)

        if summary.error:
            self.status_var.set(f"Error: {summary.error}")
            messagebox.showerror(APP_TITLE, summary.error)
            return

        verb = "Stopped" if summary.stopped_by_user else "Finished"
        msg = (f"{verb}. Processed {summary.frames} frames in {summary.duration_sec:.1f}s "
               f"(avg {summary.avg_fps:.1f} FPS).")
        if summary.save_path:
            msg += f"  Saved: {summary.save_path}"
        self.status_var.set(msg)


def enable_high_dpi() -> None:
    """Render natively on scaled displays instead of letting Windows blur-stretch the window."""
    if not sys.platform.startswith("win"):
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # System DPI aware (Win 8.1+)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # Vista / Win 7 fallback
        except (AttributeError, OSError):
            pass


def launch() -> None:
    enable_high_dpi()  # Must run before the Tk root is created
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()
