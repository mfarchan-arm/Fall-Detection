"""
========================================================================================
  AI PERSON & ELDERLY FALL DETECTION SYSTEM (GUI LAUNCHER)
  Opens the desktop GUI (gui/) which drives the detection logic (app/):
    - Live Webcam
    - Custom Video File (auto-lists videos in folder or browse any file)
    - Synthetic Occlusion Fall Demo
  Use --source to bypass the GUI and run the pipeline directly.
========================================================================================
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="AI Fall Detection System")
    parser.add_argument("--source", type=str, default=None, help="Direct source bypass ('0', 'video.mp4', 'demo')")
    args = parser.parse_args()

    if args.source is not None:
        from fall_detection_all_in_one import run_pipeline
        run_pipeline(
            source=args.source,
            save_path="fall_detection_output.mp4",
            show=True,
            conf=0.35
        )
        return

    from gui import launch
    launch()


if __name__ == "__main__":
    main()
