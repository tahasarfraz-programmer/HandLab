"""Shared helpers for HandLab (geometry, drawing, HUD)."""
import math
import time
from collections import deque

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
TOP = 34  # pixels reserved at the top for the tab strip drawn by main.py

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# ---------------------------------------------------------------- geometry
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def palm_size(lm):
    """Wrist -> middle-finger knuckle. Used to make everything scale-invariant."""
    return dist(lm[0], lm[9]) + 1e-6


def pinch_ratio(lm):
    """Thumb-tip to index-tip distance relative to palm size (small = pinching)."""
    return dist(lm[4], lm[8]) / palm_size(lm)


def pinch_point(lm):
    return ((lm[4][0] + lm[8][0]) / 2, (lm[4][1] + lm[8][1]) / 2)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def hand_bbox(lm, pad, shape):
    h, w = shape[:2]
    xs = [p[0] for p in lm]
    ys = [p[1] for p in lm]
    return (int(max(min(xs) - pad, 0)), int(max(min(ys) - pad, 0)),
            int(min(max(xs) + pad, w - 1)), int(min(max(ys) + pad, h - 1)))


# ----------------------------------------------------------------- drawing
def _ip(p):
    return int(p[0]), int(p[1])


def text(img, s, org, scale=0.55, color=(255, 255, 255), thick=1, shadow=True):
    x, y = int(org[0]), int(org[1])
    if shadow:
        cv2.putText(img, s, (x + 1, y + 1), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def panel(img, x, y, w, h, alpha=0.6):
    """Translucent dark rectangle (safe if partly off-screen)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(int(x), 0), max(int(y), 0), min(int(x + w), W), min(int(y + h), H)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.zeros_like(roi), alpha, 0)


def bar(img, x, y, w, h, frac, color=(0, 220, 0)):
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
    fw = int((w - 2) * min(max(frac, 0.0), 1.0))
    if fw > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fw, y + h - 1), color, -1)


def draw_skeleton(img, lm, line=(255, 255, 255), dot=(0, 0, 255), thick=1):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, _ip(lm[a]), _ip(lm[b]), line, thick, cv2.LINE_AA)
    for p in lm:
        cv2.circle(img, _ip(p), 3, dot, -1, cv2.LINE_AA)


class FpsCounter:
    def __init__(self, n=30):
        self.t = deque(maxlen=n)

    def tick(self):
        self.t.append(time.perf_counter())
        if len(self.t) < 2:
            return 0.0
        return (len(self.t) - 1) / (self.t[-1] - self.t[0] + 1e-9)
