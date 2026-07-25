"""
step3_basic_detection.py

Builds on step2: loads the regions you selected (rois.json) and, for each
frame, checks each region for a specific color instead of just brightness
-- turn arrows are green, the Autopilot icon is blue. For each region we
compute what fraction of its pixels fall inside that color's hue range
(with a saturation/value floor so we don't match grayish background
noise), and call it "lit" once enough of the box matches. Still no
blink-pattern logic or adaptive baseline yet -- just proving color
matching works before we add anything smarter on top.

The Autopilot icon has two possible positions (autopilot_1, autopilot_2);
either one reading "lit" counts as Autopilot being shown.

Controls:
    SPACE       - pause / resume
    N           - step forward one frame (while paused)
    B           - step backward one frame (while paused)
    drag slider - scrub to any point in the clip (auto-pauses)
    Q/ESC       - quit

Usage:
    python step3_basic_detection.py --video my_footage.mp4 --rois rois.json
    python step3_basic_detection.py --video my_footage.mp4 --rois rois.json --color-fraction 0.15
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque

import cv2
import numpy as np

BOX_COLORS = {
    "left_arrow": (0, 255, 0),
    "right_arrow": (0, 255, 255),
    "autopilot_1": (255, 0, 0),
    "autopilot_2": (255, 0, 255),
}

# Which target color each region is looking for, and the HSV hue range
# (OpenCV's 0-179 hue scale) plus minimum saturation/value that counts as
# "that color" rather than gray background noise. Widen/shift these if your
# footage's actual green/blue lands outside the defaults -- easiest to tune
# by watching the printed hue value while pointing at a known-lit icon.
COLOR_RANGES = {
    "green": {"hue": (40, 85), "min_sat": 60, "min_val": 60},
    "blue": {"hue": (95, 135), "min_sat": 60, "min_val": 60},
}

REGION_TARGET_COLOR = {
    "left_arrow": "green",
    "right_arrow": "green",
    "autopilot_1": "blue",
    "autopilot_2": "blue",
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


def color_fraction(frame, roi: dict, color_name: str) -> float:
    """Fraction (0-1) of pixels in the ROI matching the target hue range."""
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return 0.0
    spec = COLOR_RANGES[color_name]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array([spec["hue"][0], spec["min_sat"], spec["min_val"]])
    upper = np.array([spec["hue"][1], 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask)) / mask.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to a .mov or .mp4 file")
    parser.add_argument("--rois", default="rois.json", help="Regions saved by step2_select_rois.py")
    parser.add_argument("--color-fraction", type=float, default=0.15,
                         help="Minimum fraction (0-1) of a region that must match its target color to call it 'lit'")
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
          f"Color fraction threshold={args.color_fraction}. "
          f"Space=pause, N/B=step fwd/back, drag slider to scrub, Q/ESC=quit.")

    window_name = "step3: basic threshold detection"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("frame", window_name, 0, max(total_frames - 1, 1), lambda v: None)

    paused = False
    frame_index = 0
    frame = None
    fps_counter = FpsCounter()
    achieved_fps = 0.0
    last_synced_position = 0  # trackbar position we last set programmatically

    def seek_to(target_frame: int):
        """Move playback to `target_frame` (0-based) and read that frame."""
        nonlocal frame, frame_index, last_synced_position
        target_frame = max(0, min(target_frame, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, new_frame = cap.read()
        if ok:
            frame = new_frame
            frame_index = target_frame + 1
        cv2.setTrackbarPos("frame", window_name, min(frame_index, total_frames - 1))
        last_synced_position = min(frame_index, total_frames - 1)

    while True:
        loop_start = time.monotonic()

        slider_position = cv2.getTrackbarPos("frame", window_name)
        if slider_position != last_synced_position:
            # user dragged the slider -- treat as a scrub/seek request
            paused = True
            seek_to(slider_position)
        elif not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break
            frame_index += 1
            achieved_fps = fps_counter.tick()
            cv2.setTrackbarPos("frame", window_name, min(frame_index, total_frames - 1))
            last_synced_position = min(frame_index, total_frames - 1)

        display = frame.copy()
        readings = {}
        for name, roi in rois.items():
            target_color = REGION_TARGET_COLOR.get(name, "green")
            value = color_fraction(frame, roi, target_color)
            lit = value > args.color_fraction
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
            text = f"{name}: {value * 100:.0f}%  {'LIT' if lit else '-'}"
            color = (0, 255, 0) if lit else (180, 180, 180)
            cv2.putText(display, text, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            y_off += 20
        cv2.putText(display, f"AUTOPILOT: {'ON' if autopilot_lit else 'off'}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 0, 0) if autopilot_lit else (150, 150, 150), 2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n') and paused:
            seek_to(frame_index)  # frame_index is already "next frame to read" - 1 off by design of seek_to
        elif key == ord('b') and paused:
            seek_to(frame_index - 2)  # -2: undo the current frame and step one further back

        if not paused:
            elapsed = time.monotonic() - loop_start
            remaining = frame_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
