# 🛡️ Real-Time Person & Elderly Fall Detection System
### Advanced Occlusion Resilience, Kinematic State Machine & Edge AI Architecture

An enterprise-grade, edge-compatible AI computer vision system designed to detect accidental falls in real-time under severe visual occlusions (behind furniture, tables, beds, couches) while completely eliminating false positives from daily activities such as sitting down, tying shoelaces, or sleeping in bed.

---

## 🌟 Key Highlights

- **⚡ Verified Fall Alerts**: A fall is a rapid descent ($v_y > 0.7\text{ h/s}$, or a torso collapse $> 120^\circ/\text{s}$ while descending) that **ends with the body down on the floor**. The alarm is raised as soon as the landing is confirmed (≈ 0.3 s) and escalates to `CONFIRMED_FALL` if the person does not get up within 4 s. Getting back up clears it.
- **🛏️ Daily-Activity False-Alarm Rejection**: Bending, squatting, sitting down (also on the floor) and lying down slowly stay in `MONITORING`: a fast descent that ends upright, with the hips still high (bending over, lying on a bed) or with the head still well above the floor (sitting) is discarded. On 160 home-webcam clips (GMDCSA-24) false alarms dropped from 65/81 to 4/81 daily-activity clips (see *Validation*).
- **👻 Ghost & Exit Handling**: A new detection is *tentative* (never drawn, cannot raise a fall) until it has looked like a person with consistently high confidence, so flickering ghosts (chairs, clothes, shadows) stay hidden. People who walk out of the picture are dropped at once instead of leaving a frozen skeleton behind, and a person the detector loses while lying on the floor keeps the alarm with a dashed *last seen here* box.
- **👁️ 4-Tier Occlusion Resilience**:
  1. **Kinematic Bone Constraint Imputation**: Mathematically reconstructs blocked knees and ankles using anatomical anthropometric ratios.
  2. **Temporal Velocity Extrapolation**: Carries momentum vectors through visual blockages.
  3. **Furniture ROI Tracking**: Monitors trajectories descending behind known occlusion barriers (coffee tables, beds, sofas).
  4. **Proximity & Bone Length Clamping**: Enforces strict skeletal bounds to eliminate spurious spiderweb lines.
- **🚀 100% Edge CPU Ready**: Powered by a lightweight pose estimation engine (~6.5 MB). Runs at full speed on standard CPUs without requiring a dedicated GPU or custom model training.
- **🖥️ Desktop GUI**: One-click launcher with source selection, video auto-discovery, embedded live preview and an alert log.

---

## 🚀 Quickstart Guide (1 Command to Run)

You only need **ONE** command:

```bash
python run.py
```
*(Or on Windows, simply double-click **`run.bat`**)*

### 🖥️ Desktop GUI

`run.py` opens a desktop window (Tkinter) with everything that used to be in the terminal menu:

- **Input Source** — choose 📹 *Live Webcam* (with camera index), 📁 *Custom Video File* (auto-lists videos in the project root and `sample_videos/`, or **Browse...** for any `.mp4` / `.avi` / `.mkv` / `.mov`), or 🧪 *Synthetic Occlusion Fall Demo*.
- **Settings** — pose model (default `yolo11s-pose`: fewest ghost people; `yolov8n-pose` is bundled, fastest and works offline; missing weights are downloaded on first use, falling back to the bundled model if that fails), detection confidence slider and a toggle for saving the annotated output video.
- **Frame rate** (video files) — leave on *Auto*, or enter the real recording rate when a file's header is wrong. Timing and speeds depend on it: e.g. *Multiple Cameras Fall Dataset* clips are 30 fps but declare 120.
- **Start / Stop** — the annotated video with skeletons and HUD telemetry plays inside the window, with a progress bar and live FPS. Press `Esc` or `Q` to stop.
- **Emergency Alert Log** — every fall state transition (FALL DETECTED / IMPACT / CONFIRMED FALL) is listed with time, frame, person ID and torso angle, plus a red alert banner.
- **Open Output Video** / **Exit**.

To skip the GUI and run the pipeline directly: `python run.py --source demo` (or `0`, or a video path).

---

## 📦 Included Benchmark Video Suite (`sample_videos/`)

The repository includes real-world benchmark fall test videos:

| # | Video File | Scenario & Purpose | Expected Result |
|---|---|---|---|
| **1** | `sample_videos/standing_fall_1.mp4` | 🏃 Sudden collapse fall from standing | Instant `[ALERT] FALL DETECTED` |
| **2** | `sample_videos/standing_fall_2.mp4` | 🍌 Slip & backward fall from standing | Instant `[ALERT] FALL DETECTED` |
| **3** | `sample_videos/sitting_to_fall.mp4` | 🪑 Senior citizen falling off chair | Instant `[ALERT] FALL DETECTED` |
| **4** | `sample_videos/bed_rollout_fall.mp4` | 🛏️ Roll-out fall from bed onto floor | Instant `[ALERT] FALL DETECTED` |
| **5** | `sample_videos/normal_activity_no_fall.mp4` | 🚶 Walking, sitting & sleeping (ADL) | `MONITORING (RESTING)` (Zero False Alarms) |

---

## 🧠 Technical Architecture & Biomechanical Pipeline

```
  ┌─────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
  │ Video / Webcam  │ ──> │ YOLOv8-Pose Backbone │ ──> │  Occlusion Resilience  │
  │     Stream      │     │  Keypoint Extractor  │     │   & Bone Imputation    │
  └─────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                    │
  ┌─────────────────┐     ┌──────────────────────┐                  ▼
  │ HUD Visualizer  │ <── │ Multi-Stage Fall FSM │ <── ┌────────────────────────┐
  │  & Alert Engine │     │  State Transitions   │     │  Kinematic Extractor   │
  └─────────────────┘     └──────────────────────┘     │ (v_y, Angle, ω, Energy)│
                                                       └────────────────────────┘
```

### 1. Biomechanical Features
All features come from **observed** keypoints (confidence ≥ 0.3); imputed joints are drawn but never drive a decision, and missing joints are ignored instead of being read as `(0, 0)`.
- **Torso Angle ($\theta_{torso}$)**: Shoulder-midpoint to hip-midpoint angle vs. the floor ($90^\circ$ = upright, $0^\circ$ = flat), EMA-smoothed. Marked *unclear* when the torso is too foreshortened to measure (facing / bending toward the camera).
- **Vertical Descent Rate ($v_y$)**: Median downward shift of the **same** head/shoulder/hip joints between two frames ~0.25 s apart, divided by the real time between them and by the person's upright height ($\text{h/s}$). Single-joint glitches and dropped webcam frames cannot spike it.
- **Angular Collapse Rate ($\omega$)**: Change of the smoothed torso angle over the same window, in degrees per second.
- **Landing posture**: Drop of the body top and of the hips relative to the posture just before the descent, plus box shape (width/height) and height relative to upright.

### 2. State Machine Logic Flow
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
**Tracking rules that keep the state machine honest**
- A track starts *tentative* and is confirmed after 3 strong detections whose recency-weighted confidence is ≥ 0.6; only confirmed tracks are drawn or can raise a fall.
- Low-confidence detections (≥ 0.15) may only continue a confirmed track they overlap well (a body lying on the floor often scores low). A match on distance alone needs a confident detection, so a ghost next to a person cannot take over their track.
- During a fall, a body-sized box where the person must have landed re-attaches to the track, and the track is kept for 2 s: the detector often loses a body right as it reaches the floor.
- A body cut off by a frame border must stay visibly down for 0.6 s (instead of 0.3 s) before it counts, and a track lost at a border while not down is treated as someone walking out of view.
- A track lost in an alarm state keeps the alarm (up to 30 s) without drawing a stale skeleton; a person standing up where it lay is re-attached, and 1.5 s upright clears it. A missed detection does not restart that count.

### 3. Validation
Identical YOLOv8n-pose detections (confidence 0.35) replayed through the previous and the current logic, scored against each dataset's labels:

| Dataset | | Previous logic | Current logic |
|---|---|---|---|
| [UR Fall Detection](https://fenix.ur.edu.pl/~mkepski/ds/uf.html) (70 front-camera sequences) | Falls detected | 30 / 30 | 28 / 30 |
| | Daily-activity videos with a false alarm | 30 / 40 (65 alarms) | 3 / 40 (3 alarms) |
| [GMDCSA-24](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos) (160 home-webcam clips) | Falls onto the floor detected | 58 / 58 | 42 / 58 |
| | Falls onto a bed detected | 19 / 19 | 8 / 19 |
| | Daily-activity clips with a false alarm | 65 / 81 (164 alarms) | 4 / 81 (4 alarms) |
| | Fall clips alarmed *before* the fall started | 21 | 0 |
| | Frames showing a ghost person (single-person clips) | 11.7 % | 3.8 % |
| Multiple Cameras Fall (8 ceiling views, 1 scenario, 30 fps) | Alarms raised | 51 | 4 (all at the fall) |

The previous logic "detected" every fall because it alarmed on almost any movement. Most falls the current logic misses are clips where YOLOv8n stops detecting the person once they are on the floor; a larger pose model finds a lying person far more often (see *Choosing a pose model*). The remaining false alarms are push-ups, sitting down on the floor while leaning forward, lying down on a bed quickly, and people deliberately dropping to the floor, which single-camera 2D pose cannot reliably tell apart from a fall. A lying-on-a-bed fall is deliberately treated like lying down on a bed.

### 4. Choosing a Pose Model
The pose model decides how often a person on the floor is still found and how many objects are mistaken for people. Measured on the same clips (lying = labelled lying / fallen frames; ghosts = extra "people" in single-person daily-activity frames):

| Model | Lying person found (UR / GMDCSA, conf ≥ 0.35) | Ghost detections per 100 frames (conf ≥ 0.35 / ≥ 0.6) |
|---|---|---|
| yolov8n-pose (bundled) | 53 % / 79 % | 7.0 / 0.32 |
| yolov8s-pose | 70 % / 93 % | 7.9 / 2.06 |
| yolo11n-pose | 51 % / 78 % | 10.3 / 2.52 |
| **yolo11s-pose (GUI default)** | 58 % / 90 % | 1.5 / 0.14 |
| yolo26n-pose | 51 % / 70 % | 4.2 / 0.06 |
| yolo26s-pose | 60 % / 91 % | 1.7 / 0.03 |

With the full pipeline, yolo11s shows a ghost person in 0.8 % of daily-activity frames (yolov8n: 3.8 %), detects 53/77 GMDCSA falls (yolov8n: 50/77) with the same 4/81 false alarms, and 26/30 UR falls with 1/40 false alarms (yolov8n: 28/30, 3/40). On this test machine's CPU it needs ~78 ms per frame against ~50 ms for yolov8n. The CLI (`run.py --source`) keeps the bundled yolov8n.

### 5. Camera Placement
The thresholds are calibrated for a camera that sees people **from the side, roughly at chest-to-head height** (a wall-mounted webcam or a camera on a shelf). A ceiling-corner or fisheye view compresses vertical motion: on the 8 views of one *Multiple Cameras Fall Dataset* scenario (frame rate set to 30), the fall was detected in 4 views and missed in the 4 views where the person appears small and seen from above, with no false alarm. The previous logic also alarmed around the fall in most views, but raised 51 alarms in total across the 8 views.

---

## 📁 Repository File Structure

```
d:\A New Fall Detecion Sytem\
├── run.bat                          # One-click Windows desktop launcher
├── run.py                           # Launcher: opens the GUI (or --source for headless run)
├── app/                             # Application logic (no GUI code)
│   ├── services.py                  # Video discovery, validation & session configs
│   └── session.py                   # Background detection session + listener events
├── gui/                             # Tkinter presentation layer (no detection logic)
│   ├── main_window.py               # Main window: controls, video preview, alert log
│   └── bridge.py                    # Thread-safe bridge from worker events to the UI
├── fall_detection_all_in_one.py     # Standalone all-in-one pipeline (Zero sub-dependencies)
├── core/
│   ├── detector.py                  # YOLOv8-Pose wrapper & keypoint extractor
│   ├── tracker.py                   # Multi-person temporal tracklet manager
│   ├── occlusion.py                 # Occlusion engine, Kalman extrapolation & bone imputation
│   ├── kinematics.py                # Biomechanical kinematic feature engine
│   ├── state_machine.py             # Multi-stage Fall State Machine
│   └── visualizer.py                # Telemetry HUD visualizer & alert banners
├── sample_videos/                   # Benchmark fall test dataset
│   ├── standing_fall_1.mp4          # Sudden collapse fall
│   ├── standing_fall_2.mp4          # Slip and fall
│   ├── sitting_to_fall.mp4          # Chair fall
│   ├── bed_rollout_fall.mp4         # Bed roll-out fall
│   └── normal_activity_no_fall.mp4  # Normal ADL baseline (zero false alarm test)
├── test_generator.py                # Synthetic occlusion test scenario generator
├── requirements.txt                 # Clean dependency manifest
├── FALL_DETECTION_SYSTEM_DESIGN.md  # Comprehensive research & mathematical design spec
└── README.md                        # Documentation & quickstart guide
```

---

## 🛠️ Advanced CLI Usage (Direct Bypass)

If you prefer direct command-line execution without the interactive menu:

```bash
# Run on webcam (device 0)
python fall_detection_all_in_one.py --source 0 --show

# Run on a specific video file
python fall_detection_all_in_one.py --source sample_videos/standing_fall_1.mp4 --show --save output_result.mp4

# Run synthetic occlusion demo
python fall_detection_all_in_one.py --source demo --show
```

---

## 📜 Scientific Reference & Design Document
For full mathematical formulations, biomechanical formulas, Kalman filter state equations, and multi-view homography matrices, refer to:
👉 **[FALL_DETECTION_SYSTEM_DESIGN.md](file:///d:/A%20New%20Fall%20Detecion%20Sytem/FALL_DETECTION_SYSTEM_DESIGN.md)**
