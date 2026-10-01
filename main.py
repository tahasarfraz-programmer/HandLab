"""HandLab - three hand-gesture computer-vision projects in one app.

  1  Invisibility   pinch with both hands to open a portal where you vanish
  2  HandFrame AI   frame your face with your hands; pinch to change filters
  3  Sign Language  ASL fingerspelling -> letters -> words -> sentences

Global keys:  1 / 2 / 3 switch mode   h toggle HUD   f fullscreen   q or Esc quit
"""
import argparse
import sys
import time

import cv2

from handframe import HandFrameMode
from invisibility import InvisibilityMode
from signlang import SignMode
from tracker import HandTracker
from utils import FONT, FpsCounter, panel, text


def draw_tabs(frame, modes, cur, fps):
    w = frame.shape[1]
    panel(frame, 0, 0, w, 28, 0.75)
    x = 10
    for i, m in enumerate(modes):
        label = f"[{i + 1}] {m.name}"
        on = i == cur
        text(frame, label, (x, 19), 0.5, (0, 255, 255) if on else (170, 170, 170), 2 if on else 1, shadow=False)
        x += cv2.getTextSize(label, FONT, 0.5, 2)[0][0] + 26
    text(frame, f"{fps:.0f} FPS", (w - 80, 19), 0.5, (120, 255, 120), 1, shadow=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--mode", type=int, default=1, choices=(1, 2, 3), help="start mode")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Try --camera 1.")

    tracker = HandTracker()
    modes = [InvisibilityMode(), HandFrameMode(), SignMode()]
    cur, show_hud, fullscreen = args.mode - 1, True, False
    fps_counter, ms = FpsCounter(), 0.0
    win = "HandLab"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        frame = cv2.flip(frame, 1)                       # mirror, like a selfie
        hands = tracker.process(frame)
        fps = fps_counter.tick()
        modes[cur].update(frame, hands, show_hud, {"fps": fps, "ms": ms})
        draw_tabs(frame, modes, cur, fps)
        cv2.imshow(win, frame)
        ms = (time.perf_counter() - t0) * 1000

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == 255 or modes[cur].on_key(k):             # the active mode gets first pick of keys
            continue
        if k == ord("q"):
            break
        if k in (ord("1"), ord("2"), ord("3")):
            cur = k - ord("1")
        elif k == ord("h"):
            show_hud = not show_hud
        elif k == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
