"""
step2_select_rois.py

Builds on step1: instead of looking at the whole frame, pick out just the
small regions that actually matter -- left turn arrow, right turn arrow,
and the Autopilot/FSD icon -- and confirm they land correctly across the
whole clip. Still no brightness/color detection logic yet.

The Autopilot icon can appear in two different screen positions depending
on UI state, so it gets two boxes (autopilot_1, autopilot_2) -- either one
lighting up counts as Autopilot being shown.

Flow:
  1. Plays the video like step1. Pause on a frame where the dashboard is
     clearly visible (SPACE to pause, N to step).
  2. Press R to enter region-selection mode: drag-select the left arrow
     box, the right arrow box, then both Autopilot icon positions.
  3. Saves those boxes to rois.json.
  4. Playback continues (from where you paused) with the three boxes drawn
     as overlays on every frame, so you can watch the whole clip and
     visually confirm they stay correctly positioned over the real icons
     -- catching camera framing issues before any detection math exists.

Usage:
    python step2_select_rois.py --video my_footage.mp4
    python step2_select_rois.py --video my_footage.mp4 --rois rois.json   # reuse a saved selection
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque

import cv2

BOX_LABELS = [
    ("left_arrow", (0, 255, 0)),
    ("right_arrow", (0, 255, 255)),
    ("autopilot_1", (255, 0, 0)),
    ("autopilot_2", (255, 0, 255)),
]


class FpsCounter:
    """Tracks *actual* achieved loop rate (real wall-clock time between
    frames), not the video's nominal fps -- so you can see if adding more
    processing on top is causing real frame drops/slowdown."""

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


def select_rois(frame) -> dict:
    rois = {}
    for name, _ in BOX_LABELS:
        print(f"Drag-select the '{name}' region, then press ENTER/SPACE. (C to cancel this box)")
        box = cv2.selectROI(f"Select: {name}", frame, showCrosshair=True)
        cv2.destroyWindow(f"Select: {name}")
        x, y, w, h = [int(v) for v in box]
        if w == 0 or h == 0:
            print(f"  Skipped '{name}' (empty selection).")
            continue
        rois[name] = {"x": x, "y": y, "w": w, "h": h}
        print(f"  {name}: {rois[name]}")
    return rois


def draw_rois(frame, rois: dict):
    display = frame.copy()
    for name, color in BOX_LABELS:
        roi = rois.get(name)
        if roi is None:
            continue
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        cv2.putText(display, name, (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to a .mov or .mp4 file")
    parser.add_argument("--rois", default="rois.json", help="Where to save/load region selections")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rois = {}
    try:
        with open(args.rois) as f:
            rois = json.load(f)
        print(f"Loaded existing regions from {args.rois}: {list(rois.keys())}")
    except FileNotFoundError:
        pass

    print(f"Opened {args.video}: {fps:.1f} fps, {total_frames} frames "
          f"({total_frames / fps:.1f}s).")
    print("Space=pause, N/B=step fwd/back, drag slider to scrub, "
          "R=select regions on current frame, S=save regions, Q/ESC=quit.")

    window_name = "step2: region selection + verification"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("frame", window_name, 0, max(total_frames - 1, 1), lambda v: None)

    paused = False
    frame_index = 0
    frame = None
    fps_counter = FpsCounter()
    achieved_fps = 0.0
    last_synced_position = 0

    def seek_to(target_frame: int):
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

        display = draw_rois(frame, rois)
        cv2.putText(display, f"frame {frame_index}/{total_frames}"
                    + ("  [PAUSED]" if paused else ""),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        fps_color = (0, 255, 0) if achieved_fps >= fps * 0.95 else (0, 0, 255)
        cv2.putText(display, f"source: {fps:.1f} fps  |  actual: {achieved_fps:.1f} fps",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n') and paused:
            seek_to(frame_index)
        elif key == ord('b') and paused:
            seek_to(frame_index - 2)
        elif key == ord('r'):
            if not paused:
                paused = True
            new_rois = select_rois(frame)
            rois.update(new_rois)
        elif key == ord('s'):
            with open(args.rois, "w") as f:
                json.dump(rois, f, indent=2)
            print(f"Saved regions to {args.rois}: {rois}")

        if not paused:
            elapsed = time.monotonic() - loop_start
            remaining = frame_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()

    if rois:
        with open(args.rois, "w") as f:
            json.dump(rois, f, indent=2)
        print(f"Final regions saved to {args.rois}: {rois}")
    else:
        print("No regions were selected -- nothing saved.")


if __name__ == "__main__":
    main()
