"""
step1_playback.py

The smallest possible starting point: open a .mov or .mp4 file recorded
from the dashboard camera and play it back as if it were a live feed, one
frame at a time, at (roughly) the video's real playback speed. No
detection logic yet -- this just proves we can read your footage
correctly and lets you watch it the same way the live program eventually
will.

Usage:
    python step1_playback.py --video my_footage.mov
    python step1_playback.py --video my_footage.mp4

Controls while playing:
    SPACE       - pause / resume
    N           - step forward one frame (while paused)
    B           - step backward one frame (while paused)
    drag slider - scrub to any point in the clip (auto-pauses)
    ESC/Q       - quit
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to a .mov, .mp4 (or any OpenCV-readable) file")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Opened {args.video}: {fps:.1f} fps, {total_frames} frames "
          f"({total_frames / fps:.1f}s). Space=pause, N/B=step fwd/back, "
          f"drag slider to scrub, Q/ESC=quit.")

    window_name = "step1: simulated live feed"
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

        display = frame.copy()
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

        if not paused:
            elapsed = time.monotonic() - loop_start
            remaining = frame_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
