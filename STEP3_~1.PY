"""
step3_basic_detection.py

Builds on step2: loads the regions you selected (rois.json) and, for each
frame, checks each region for a specific color instead of just brightness
-- turn arrows are green, checked with a simple color-fraction match.

The Autopilot icon (steering yoke) is checked more specifically: a plain
blue object (a car, sky, signage) would trip a simple "is this region
blue" check, so instead we require the region to be blue-dominant AND
contain a distinctly brighter sub-shape within it (the white/light yoke
sitting on the blue background). Based on sample crops from real footage,
even the "white" yoke pixels stay fairly saturated in dark/underexposed
shots rather than reading as pure white, so the brighter-shape check is
relative to the region's own background brightness rather than an
absolute "is this pixel white" threshold. This is meaningfully harder to
false-trigger on than plain color matching. Still no blink-pattern logic
or adaptive baseline yet -- just proving this matching approach works
before adding anything smarter on top.

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

# Which target color each turn-arrow region is looking for, and the HSV hue
# range (OpenCV's 0-179 hue scale) plus minimum saturation/value that counts
# as "that color" rather than gray background noise. Widen/shift these if
# your footage's actual green lands outside the defaults -- easiest to tune
# by watching the printed hue value while pointing at a known-lit icon.
COLOR_RANGES = {
    "green": {"hue": (40, 85), "min_sat": 60, "min_val": 60},
}

ARROW_REGIONS = {"left_arrow": "green", "right_arrow": "green"}

# Autopilot yoke icon spec. Blue range below is field-calibrated against
# real nighttime footage: a high saturation floor (180) is what actually
# separates true blue from gray dashboard noise at night -- grays read as
# low-saturation regardless of brightness, so a saturation floor rules them
# out far better than a hue range alone. The value range is also capped
# (not just floored) since the calibrated blue itself stays fairly dim.
AUTOPILOT_SPEC = {
    "blue_hue": (115, 135),
    "min_sat": 180,
    "min_val": 5,
    "max_val": 112,
    "min_blue_fraction": 0.40,   # region must be mostly this blue overall
    "brightness_contrast_ratio": 1.3,  # "bright shape" must exceed this x the blue background's median brightness
    "min_bright_fraction": 0.04,       # at least this much of the region must be that much brighter (the yoke)
}

AUTOPILOT_REGIONS = ("autopilot_1", "autopilot_2")


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


def autopilot_icon_score(frame, roi: dict) -> dict:
    """
    Checks a region for the "blue background + brighter yoke shape" signature
    instead of just "is this region blue" -- a plain blue object (car, sky)
    is blue but usually lacks a distinct brighter sub-shape sitting inside it.

    Returns a dict with the raw measurements plus a final `lit` bool, so the
    caller can display why a region did or didn't match.
    """
    spec = AUTOPILOT_SPEC
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return {"blue_fraction": 0.0, "bright_fraction": 0.0, "lit": False}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    blue_mask = (
        (h_ch >= spec["blue_hue"][0]) & (h_ch <= spec["blue_hue"][1]) &
        (s_ch >= spec["min_sat"]) &
        (v_ch >= spec["min_val"]) & (v_ch <= spec["max_val"])
    )
    blue_fraction = float(np.count_nonzero(blue_mask)) / blue_mask.size

    if blue_fraction < spec["min_blue_fraction"] or not np.any(blue_mask):
        return {"blue_fraction": blue_fraction, "bright_fraction": 0.0, "lit": False}

    background_median_v = float(np.median(v_ch[blue_mask]))
    bright_threshold = background_median_v * spec["brightness_contrast_ratio"]
    bright_mask = v_ch > bright_threshold
    bright_fraction = float(np.count_nonzero(bright_mask)) / bright_mask.size

    lit = bright_fraction >= spec["min_bright_fraction"]
    return {"blue_fraction": blue_fraction, "bright_fraction": bright_fraction, "lit": lit}


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
        readings = {}       # name -> (display_value, lit) for arrows
        ap_readings = {}    # name -> autopilot_icon_score() dict for autopilot boxes
        for name, roi in rois.items():
            if name in AUTOPILOT_REGIONS:
                score = autopilot_icon_score(frame, roi)
                ap_readings[name] = score
                lit = score["lit"]
            else:
                target_color = ARROW_REGIONS.get(name, "green")
                value = color_fraction(frame, roi, target_color)
                lit = value > args.color_fraction
                readings[name] = (value, lit)

            color = BOX_COLORS.get(name, (200, 200, 200))
            box_color = (0, 255, 0) if lit else color
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2 if lit else 1)

        # When a turn signal is active, the dashboard swaps a live blind-spot
        # camera feed into the autopilot_2 position -- which can contain real
        # blue objects (a car, sky, signage) and falsely read as Autopilot.
        # The real Autopilot icon can only still be showing in autopilot_1
        # during a turn signal, so autopilot_2 is ignored while one is active.
        turn_signal_active = readings.get("left_arrow", (0, False))[1] or \
            readings.get("right_arrow", (0, False))[1]

        autopilot_1_lit = ap_readings.get("autopilot_1", {}).get("lit", False)
        autopilot_2_lit = ap_readings.get("autopilot_2", {}).get("lit", False) and not turn_signal_active
        autopilot_lit = autopilot_1_lit or autopilot_2_lit

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
        for name, score in ap_readings.items():
            if name == "autopilot_2" and turn_signal_active:
                text = f"{name}: blue={score['blue_fraction']*100:.0f}% bright={score['bright_fraction']*100:.0f}%  (ignored - turn signal active)"
                color = (100, 100, 100)
            else:
                lit = score["lit"]
                text = f"{name}: blue={score['blue_fraction']*100:.0f}% bright={score['bright_fraction']*100:.0f}%  {'LIT' if lit else '-'}"
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
