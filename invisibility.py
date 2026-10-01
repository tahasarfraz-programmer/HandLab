"""Mode 1 - Invisibility.

Capture the empty background once, then pinch (thumb + index) with BOTH hands.
The two pinch points become opposite corners of a "portal". Inside it, you
disappear and the empty background shows through.
"""
import math
import time

import cv2
import numpy as np

from utils import FONT, TOP, draw_skeleton, ease, panel, pinch_point, pinch_ratio, text

PINCH_ON = 0.30      # pinch if thumb-index gap < 30% of palm size
DIFF_THRESH = 28     # how different a pixel must be from the background to count as "you"
MIN_PORTAL = 60      # smallest portal side, in pixels


def banner(frame, msg, sub=""):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(msg, FONT, 0.9, 2)
    panel(frame, w // 2 - tw // 2 - 20, h // 2 - th - 20, tw + 40, th + 50 + (26 if sub else 0), 0.7)
    text(frame, msg, (w // 2 - tw // 2, h // 2), 0.9, (0, 255, 255), 2)
    if sub:
        (sw, _), _ = cv2.getTextSize(sub, FONT, 0.5, 1)
        text(frame, sub, (w // 2 - sw // 2, h // 2 + 30), 0.5)


class InvisibilityMode:
    name = "Invisibility"
    COUNTDOWN = 3.0
    FRAMES = 20

    def __init__(self):
        self.plate = None          # clean background image
        self.cap_start = None      # when the capture countdown began
        self.acc = None
        self.acc_n = 0
        self.progress = 0.0        # portal open amount 0..1
        self.rect = None           # last portal rectangle (x1, y1, x2, y2)
        self.k5 = np.ones((5, 5), np.uint8)

    # ------------------------------------------------------------- input
    def on_key(self, k):
        if k in (ord("b"), ord("B")):
            self.cap_start, self.acc, self.acc_n = time.time(), None, 0
            return True
        if k in (ord("r"), ord("R")):
            self.plate, self.rect, self.progress = None, None, 0.0
            return True
        return False

    # ------------------------------------------------------------ update
    def update(self, frame, hands, hud=True, ctx=None):
        h, w = frame.shape[:2]

        if self.cap_start is not None:                      # capturing the background
            left = self.COUNTDOWN - (time.time() - self.cap_start)
            if left > 0:
                banner(frame, f"STEP OUT OF FRAME  {int(math.ceil(left))}", "capturing the empty room...")
                return
            if self.acc is None:
                self.acc = np.zeros(frame.shape, np.float32)
            self.acc += frame
            self.acc_n += 1
            banner(frame, f"CAPTURING BACKGROUND {self.acc_n}/{self.FRAMES}")
            if self.acc_n >= self.FRAMES:
                self.plate = (self.acc / self.acc_n).astype(np.uint8)
                self.cap_start = None
            return

        if self.plate is None:
            banner(frame, "PRESS  B  TO CAPTURE BACKGROUND", "stand out of the frame, keep the camera still")
            return
        if self.plate.shape != frame.shape:
            self.plate = cv2.resize(self.plate, (w, h))

        # two pinching hands -> portal rectangle
        pts = [pinch_point(lm) for lm in hands if pinch_ratio(lm) < PINCH_ON]
        target = 0.0
        if len(pts) >= 2:
            (ax, ay), (bx, by) = pts[0], pts[1]
            r = np.array([min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)], float)
            if r[2] - r[0] > MIN_PORTAL and r[3] - r[1] > MIN_PORTAL:
                self.rect = r if self.rect is None else self.rect * 0.5 + r * 0.5
                target = 1.0
        self.progress += (target - self.progress) * 0.2
        if self.progress < 0.01:
            self.progress = 0.0

        if self.rect is not None and self.progress > 0.02:
            e = ease(self.progress)
            cx, cy = (self.rect[0] + self.rect[2]) / 2, (self.rect[1] + self.rect[3]) / 2
            hw, hh = (self.rect[2] - self.rect[0]) / 2 * e, (self.rect[3] - self.rect[1]) / 2 * e
            x1, y1 = int(max(cx - hw, 0)), int(max(cy - hh, 0))
            x2, y2 = int(min(cx + hw, w - 1)), int(min(cy + hh, h - 1))
            if x2 - x1 > 6 and y2 - y1 > 6:
                self._remove_person(frame, x1, y1, x2, y2)
                self._draw_portal(frame, x1, y1, x2, y2)

        for lm in hands:
            draw_skeleton(frame, lm, (255, 255, 255), (0, 0, 255), 1)

        if hud:
            if self.progress == 0:
                text(frame, "PINCH WITH BOTH HANDS -> OPEN PORTAL", (w // 2 - 190, TOP + 22), 0.6, (0, 255, 255), 2)
            panel(frame, 0, h - 26, w, 26, 0.7)
            text(frame, f"PORTAL {self.progress * 100:.0f}%     HANDS {len(hands)}     PINCH {len(pts)}/2"
                        f"     [B] re-capture background   [R] reset",
                 (12, h - 8), 0.45, (0, 255, 255), 1, shadow=False)

    # ----------------------------------------------------------- effects
    def _remove_person(self, frame, x1, y1, x2, y2):
        """Inside the portal, replace pixels that differ from the background (= you)."""
        h, w = frame.shape[:2]
        ph, pw = max(8, h // 10), max(8, w // 10)

        def corners(a):  # mean colour of the four frame corners (rarely contain you)
            return np.concatenate([a[:ph, :pw].reshape(-1, 3), a[:ph, -pw:].reshape(-1, 3),
                                   a[-ph:, :pw].reshape(-1, 3), a[-ph:, -pw:].reshape(-1, 3)]).mean(0)

        # compensate for camera auto-exposure drifting since the capture
        gain = np.clip(corners(frame) / np.maximum(corners(self.plate), 1), 0.6, 1.6)
        roi = frame[y1:y2, x1:x2]
        bg = np.clip(self.plate[y1:y2, x1:x2].astype(np.float32) * gain, 0, 255).astype(np.uint8)
        diff = cv2.absdiff(roi, bg).max(axis=2)
        m = (diff > DIFF_THRESH).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k5)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        # fill holes: clothing that happens to match the wall must not leave gaps in the silhouette
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 400]
        m = np.zeros_like(m)
        cv2.drawContours(m, cnts, -1, 255, cv2.FILLED)
        m = cv2.dilate(m, self.k5, iterations=2)
        a = (cv2.GaussianBlur(m, (15, 15), 0).astype(np.float32) / 255.0)[..., None]
        frame[y1:y2, x1:x2] = (roi * (1 - a) + bg * a).astype(np.uint8)

    def _draw_portal(self, frame, x1, y1, x2, y2):
        h, w = frame.shape[:2]
        gx0, gy0, gx1, gy1 = max(x1 - 30, 0), max(y1 - 30, 0), min(x2 + 30, w), min(y2 + 30, h)
        glow = np.zeros((gy1 - gy0, gx1 - gx0, 3), np.uint8)
        cv2.rectangle(glow, (x1 - gx0, y1 - gy0), (x2 - gx0, y2 - gy0), (0, 255, 255), 4)
        glow = cv2.GaussianBlur(glow, (0, 0), 8)
        frame[gy0:gy1, gx0:gx1] = cv2.add(frame[gy0:gy1, gx0:gx1], glow)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
        L = 22
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + dx * L, cy), (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * L), (255, 255, 255), 3, cv2.LINE_AA)
