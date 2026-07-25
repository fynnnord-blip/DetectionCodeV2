"""
step3_basic_detection.py

Builds on step2: loads the regions you selected (rois.json) and, for each
frame, computes the average brightness inside each box and compares it to
a single fixed threshold. That's it -- no blink-pattern logic, no adaptive
baseline, no color classification yet. The goal here is just to prove the
core measurement works and watch the raw numbers move in response to real
footage before we add any smarter behavior on top.

The Autopilot icon has two possible positions (autopilot_1, autopilot_2);
either one reading "lit" counts as Autopilot being shown.

Usage:
    python step3_basic_detection.py --video my_footage.mp4 --rois rois.json
    python step3_basic_detection.py --video my_footage.mp4 --rois rois.json --threshold 150
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque

import cv2

BOX_COLORS = {
    "left_arrow": (0, 255, 0),
    "right_arrow": (0, 255, 255),
    "autopilot_1": (255, 0, 0),
    "autopilot_2": (255, 0, 255),
}


class FpsCounter:
    def __init__(self, window=30):
        self._timestamps = deque(maxlen=window)

    def tick(self) -> float:
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / span


def brightness(frame, roi: dict) -> float:
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to a .mov or .mp4 file")
    parser.add_argument("--rois", default="rois.json", help="Regions saved by step2_select_rois.py")
    parser.add_argument("--threshold", type=float, default=150.0,
                         help="Fixed brightness threshold (0-255) to call a region 'lit'")
    args = parser.parse_args()

    with open(args.rois) as f:
        rois = json.load(f)
    if not rois:
        raise SystemExit(f"No regions found in {args.rois}. Run step2_select_rois.py first.")
    print(f"Loaded regions: {list(rois.keys())}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Opened {args.video}: {fps:.1f} fps, {total_frames} frames. "
          f"Threshold={args.threshold}. Space=pause, N=step, Q/ESC=quit.")

    paused = False
    frame_index = 0
    frame = None
    fps_counter = FpsCounter()
    achieved_fps = 0.0

    while True:
        loop_start = time.monotonic()

        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break
            frame_index += 1
            achieved_fps = fps_counter.tick()

        display = frame.copy()
        readings = {}
        for name, roi in rois.items():
            value = brightness(frame, roi)
            lit = value > args.threshold
            readings[name] = (value, lit)

            color = BOX_COLORS.get(name, (200, 200, 200))
            box_color = (0, 255, 0) if lit else color
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2 if lit else 1)

        autopilot_lit = readings.get("autopilot_1", (0, False))[1] or \
            readings.get("autopilot_2", (0, False))[1]

        y_off = 25
        cv2.putText(display, f"frame {frame_index}/{total_frames}"
                    + ("  [PAUSED]" if paused else ""),
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_off += 25
        fps_color = (0, 255, 0) if achieved_fps >= fps * 0.95 else (0, 0, 255)
        cv2.putText(display, f"source: {fps:.1f} fps | actual: {achieved_fps:.1f} fps",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        y_off += 25
        for name, (value, lit) in readings.items():
            text = f"{name}: {value:.0f}  {'LIT' if lit else '-'}"
            color = (0, 255, 0) if lit else (180, 180, 180)
            cv2.putText(display, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y_off += 20
        cv2.putText(display, f"AUTOPILOT: {'ON' if autopilot_lit else 'off'}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0) if autopilot_lit else (150, 150, 150), 2)

        cv2.imshow("step3: basic threshold detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n') and paused:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break
            frame_index += 1

        if not paused:
            elapsed = time.monotonic() - loop_start
            remaining = frame_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
