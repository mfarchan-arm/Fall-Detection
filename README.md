# 🛡️ Fall Detection
### Real-time person & elderly fall detection with YOLO pose, a kinematic state machine and a desktop GUI

Fall Detection watches a webcam or a video file, tracks each person's skeleton with a YOLO pose model, and raises an alarm when someone **falls and stays down on the floor**. It runs on a normal CPU, needs no model training, and is built to keep false alarms from daily activities (sitting, bending, lying down on a bed, walking out of the picture) low.

---

## 🌟 Highlights

- **⚡ Verified fall alerts** — a fall is a rapid descent ($v_y > 0.7\text{ h/s}$, or a torso collapse $> 120^\circ/\text{s}$ while descending) that **ends with the body down on the floor**. The alarm is raised as soon as the landing is confirmed (≈ 0.3 s) and escalates to `CONFIRMED_FALL` if the person does not get up within 4 s. Getting back up clears it.
- **🛏️ Daily-activity rejection** — bending, squatting, sitting down (also on the floor) and lying down slowly stay in `MONITORING`. On 160 home-webcam clips (GMDCSA-24) false alarms dropped from 65/81 to 4/81 daily-activity clips compared with the original logic (see [Validation](#3-validation)).
- **👻 Ghost & exit handling** — a new detection stays *tentative* (not drawn, cannot raise a fall) until it has looked like a person with consistently high confidence, so chairs, hanging clothes and shadows stay hidden. People who walk out of the picture are dropped at once instead of leaving a frozen skeleton behind.
- **👁️ Occlusion handling** — hidden joints are estimated from bone proportions and recent motion for display; a person the detector loses while lying on the floor keeps the alarm with a dashed *last seen here* box; optional furniture zones help with falls behind tables or beds.
- **🖥️ Desktop GUI** — webcam / video / demo source selection, live annotated preview, alert log, choice of pose model. GUI code (`gui/`) and application logic (`app/`) are kept separate.

---

## 🚀 Getting Started

**Requirements:** Python 3.9+ and a CPU (a CUDA GPU is used automatically if PyTorch sees one).

```bash
git clone https://github.com/mfarchan-arm/Fall-Detection.git
cd Fall-Detection

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt

python run.py
```

On Windows you can also double-click **`run.bat`** (it uses `.venv` when present).

### 🖥️ Using the GUI

- **Input Source** — 📹 *Live Webcam* (choose the camera index), 📁 *Custom Video File* (lists videos in the project folder and `sample_videos/`, or **Browse...** for any `.mp4` / `.avi` / `.mkv` / `.mov`), or 🧪 *Synthetic Occlusion Fall Demo*.
- **Frame rate** (video files) — leave on *Auto*, or enter the real recording rate when a file's header is wrong. Speeds and timers depend on it (e.g. *Multiple Cameras Fall Dataset* clips are 30 fps but declare 120).
- **Pose model** — default `yolo11s-pose` (fewest ghost people); `yolov8n-pose` is bundled, fastest and works offline. Missing weights are downloaded on first use; if that fails the bundled model is used. See [Choosing a pose model](#4-choosing-a-pose-model).
- **Detection confidence** and **Save annotated output video**.
- **Start / Stop** — the annotated video plays inside the window with a progress bar and FPS. `Esc` or `Q` stops.
- **Emergency Alert Log** — every `FALL DETECTED` / `CONFIRMED FALL` with time, frame, person ID and torso angle, plus a red banner.

### ⌨️ Command line (no GUI)

```bash
python run.py --source demo              # synthetic demo
python run.py --source 0                 # webcam 0
python run.py --source path/to/video.mp4

# More options (bundled yolov8n model):
python fall_detection_all_in_one.py --source path/to/video.mp4 --show --save result.mp4 --conf 0.35
```

---

## 🎬 Test Videos

Videos are not stored in the repository (`*.mp4`, `*.avi` and `sample_videos/` are git-ignored). Put your own clips in `sample_videos/` and they appear in the GUI list. Public datasets used for validation:

| Dataset | What it contains |
|---|---|
| [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) | 30 falls + 40 daily activities, front camera with per-frame labels. The `cam0` files show depth (left) and RGB (right) side by side; crop the right half. Non-commercial academic use. |
| [GMDCSA-24](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos) | 79 falls + 81 daily activities recorded with a webcam in real homes, day and night (CC BY 4.0). |
| Multiple Cameras Fall Dataset | 24 scenarios × 8 ceiling-corner fisheye cameras. Set the GUI frame rate to **30**. |

---

## 🧠 How It Works

```
  ┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
  │ Video / Webcam  │ ──> │   YOLO Pose Model    │ ──> │  Tracker (tentative →  │
  │     Stream      │     │  17 COCO keypoints   │     │  confirmed, exits)     │
  └─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                    │
  ┌─────────────────┐     ┌──────────────────────┐                  ▼
  │ HUD Visualizer  │ <── │   Fall State Machine │ <── ┌────────────────────────┐
  │  & Alert Log    │     │                      │     │  Kinematic Features    │
  └─────────────────┘     └──────────────────────┘     │ (v_y, angle, ω, pose)  │
                                                       └────────────────────────┘
```

Webcam frames are read on their own thread and only the newest frame is analysed, so a slow CPU never falls behind real time; timings use real capture timestamps.

### 1. Biomechanical Features
All features come from **observed** keypoints (confidence ≥ 0.3); estimated joints are drawn but never drive a decision, and missing joints are ignored instead of being read as `(0, 0)`.
- **Torso angle ($\theta_{torso}$)** — shoulder-midpoint to hip-midpoint angle vs. the floor ($90^\circ$ upright, $0^\circ$ flat), smoothed. Marked *unclear* when the torso is too foreshortened to measure.
- **Vertical descent rate ($v_y$)** — median downward shift of the **same** head/shoulder/hip joints over ~0.25 s, divided by the elapsed time and by the person's upright height (h/s). Single-joint glitches and dropped frames cannot spike it.
- **Angular collapse rate ($\omega$)** — change of the smoothed torso angle over the same window, in degrees per second.
- **Landing posture** — drop of the body top and of the hips compared with the posture just before the descent, box shape, and whether the torso is upright (sitting) or flat (lying).

### 2. State Machine
```
 [ MONITORING ] --( fast descent sustained >= 0.1 s )--> [ RAPID_DESCENT ]   (no alarm yet)
       ^                                                   |          |
       |          ( ends upright within 2 s:               |          | ( down on the floor >= 0.3 s:
       +------------ bend / sit / squat ) -----------------+          |   head AND hips dropped )
       |                                                              v
       |                                                    [ FALL_DETECTED ]   (alarm)
       |                                                              | ( not up after 4 s )
       |                                                              v
       +---------------( upright for 1.5 s )--------------- [ CONFIRMED_FALL ]  (emergency)
```

**Tracking rules**
- A track starts *tentative* and is confirmed after 3 strong detections whose recency-weighted confidence is ≥ 0.6; only confirmed tracks are drawn or can raise a fall.
- Low-confidence detections (≥ 0.15) may only continue a confirmed track they overlap well (a body lying on the floor often scores low). A match on distance alone needs a confident detection, so a ghost next to a person cannot take over their track.
- During a fall, a body-sized box where the person must have landed re-attaches to the track, and the track is kept for 2 s, because the detector often loses a body right as it reaches the floor.
- A body cut off by a frame border must stay visibly down for 0.6 s before it counts, and a track lost at a border while not down is treated as someone walking out of view.
- A track lost in an alarm state keeps the alarm (up to 30 s) without drawing a stale skeleton; a person standing up where it lay is re-attached, and 1.5 s upright clears it.

### 3. Validation
Identical YOLOv8n-pose detections (confidence 0.35) replayed through the original logic and the current logic, scored against each dataset's labels:

| Dataset | | Original logic | Current logic |
|---|---|---|---|
| UR Fall Detection (70 sequences) | Falls detected | 30 / 30 | 28 / 30 |
| | Daily-activity videos with a false alarm | 30 / 40 (65 alarms) | 3 / 40 (3 alarms) |
| GMDCSA-24 (160 home-webcam clips) | Falls onto the floor detected | 58 / 58 | 42 / 58 |
| | Falls onto a bed detected | 19 / 19 | 8 / 19 |
| | Daily-activity clips with a false alarm | 65 / 81 (164 alarms) | 4 / 81 (4 alarms) |
| | Fall clips alarmed *before* the fall started | 21 | 0 |
| | Frames showing a ghost person | 11.7 % | 3.8 % |
| Multiple Cameras Fall (8 views, 1 scenario) | Alarms raised | 51 | 4 (all at the fall) |

The original logic "detected" every fall because it alarmed on almost any movement. Most falls the current logic misses are clips where the nano model stops detecting the person once they are on the floor; a larger pose model helps (next section).

**Known limitations:** push-ups, sitting on the floor while leaning forward, lying down on a bed quickly and deliberately dropping to the floor can still raise an alarm; falls onto a bed are treated like lying down on a bed; a single 2D camera cannot always tell these apart from a real fall.

### 4. Choosing a Pose Model
How often a person on the floor is still found, and how many objects are mistaken for people (same clips for every model):

| Model | Lying person found (UR / GMDCSA, conf ≥ 0.35) | Ghost detections per 100 frames (conf ≥ 0.35 / ≥ 0.6) |
|---|---|---|
| yolov8n-pose (bundled) | 53 % / 79 % | 7.0 / 0.32 |
| yolov8s-pose | 70 % / 93 % | 7.9 / 2.06 |
| yolo11n-pose | 51 % / 78 % | 10.3 / 2.52 |
| **yolo11s-pose (GUI default)** | 58 % / 90 % | 1.5 / 0.14 |
| yolo26n-pose | 51 % / 70 % | 4.2 / 0.06 |
| yolo26s-pose | 60 % / 91 % | 1.7 / 0.03 |

With the full pipeline, yolo11s shows a ghost person in 0.8 % of daily-activity frames (yolov8n: 3.8 %), detects 53/77 GMDCSA falls (yolov8n: 50/77) with the same 4/81 false alarms, and 26/30 UR falls with 1/40 false alarms (yolov8n: 28/30, 3/40). On the test CPU it needs ~78 ms per frame against ~50 ms for yolov8n.

### 5. Camera Placement
Thresholds are calibrated for a camera that sees people **from the side, roughly at chest-to-head height** (a wall-mounted webcam or a camera on a shelf). Ceiling-corner or fisheye views compress vertical motion: on the 8 views of one Multiple Cameras Fall scenario the fall was detected in 4 views, and missed where the person appears small and seen from above.

---

## 📁 Project Structure

```
Fall-Detection/
├── run.py                           # Launcher: opens the GUI (or --source for a headless run)
├── run.bat                          # Windows launcher (prefers .venv)
├── app/                             # Application logic, no GUI code
│   ├── services.py                  # Video discovery, pose models, session configs
│   └── session.py                   # Background detection session (webcam reader, video writer)
├── gui/                             # Tkinter presentation layer, no detection logic
│   ├── main_window.py               # Main window: controls, video preview, alert log
│   └── bridge.py                    # Thread-safe bridge from worker events to the UI
├── fall_detection_all_in_one.py     # Detection pipeline: detector, tracker, features, state machine, HUD
├── test_generator.py                # Synthetic occlusion test video generator
├── core/, main.py                   # Earlier modular version of the pipeline (not used by the GUI)
├── yolov8n-pose.pt                  # Bundled pose model
├── requirements.txt
├── FALL_DETECTION_SYSTEM_DESIGN.md  # Research & design notes
└── SECURITY.md
```

---

## 📜 Design Document
For the original mathematical formulation and design notes, see **[FALL_DETECTION_SYSTEM_DESIGN.md](FALL_DETECTION_SYSTEM_DESIGN.md)**. The validated behaviour of the current code is the one described in this README.
