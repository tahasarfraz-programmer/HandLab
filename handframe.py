"""Mode 2 - HandFrame AI.

Make a frame with both hands (thumb + index of each hand are the four corners).
A live, stylised copy of the camera floats inside it.

  quick pinch        -> next filter
  hold pinch 0.6 s   -> "AI render": freeze a snapshot and paint it
"""
import threading
import time
from collections import deque

import cv2
import numpy as np

from utils import TOP, panel, pinch_ratio, text

PINCH_ON = 0.30
HOLD_SECONDS = 0.6
WORK_WIDTH = 640     # filters run on a downscaled frame for speed


# ----------------------------------------------------------------- styles
def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def f_thermal(img):
    g = cv2.createCLAHE(2.0, (8, 8)).apply(_gray(img))
    out = cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)
    rows = out[::32, :].astype(np.int16) * 0.6 + 80
    cols = out[:, ::32].astype(np.int16) * 0.6 + 80
    out[::32, :] = np.clip(rows, 0, 255).astype(np.uint8)
    out[:, ::32] = np.clip(cols, 0, 255).astype(np.uint8)
    return out


def f_neon(img):
    e = cv2.dilate(cv2.Canny(cv2.GaussianBlur(_gray(img), (5, 5), 0), 50, 130), None)
    h, w = e.shape
    xs = np.linspace(0, 1, w, dtype=np.float32)
    grad = np.zeros_like(img)
    grad[..., 0] = 255
    grad[..., 1] = (255 * xs)[None, :]
    grad[..., 2] = (255 * (1 - xs))[None, :]
    neon = cv2.bitwise_and(grad, grad, mask=e)
    glow = cv2.GaussianBlur(neon, (0, 0), 6)
    return cv2.add(cv2.add((img * 0.2).astype(np.uint8), neon), glow)


def f_sketch(img):
    g = _gray(img)
    blur = cv2.GaussianBlur(255 - g, (21, 21), 0)
    return cv2.cvtColor(cv2.divide(g, 255 - blur, scale=256), cv2.COLOR_GRAY2BGR)


def f_comic(img):
    smooth = cv2.bilateralFilter(cv2.bilateralFilter(img, 9, 60, 60), 9, 60, 60)
    edges = cv2.adaptiveThreshold(cv2.medianBlur(_gray(img), 5), 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7)
    q = (smooth // 48) * 48 + 24
    return cv2.bitwise_and(q, q, mask=edges)


def f_pixel(img):
    h, w = img.shape[:2]
    s = cv2.resize(img, (max(w // 14, 1), max(h // 14, 1)), interpolation=cv2.INTER_AREA)
    s = (s // 48) * 48 + 24
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)


def f_glitch(img):
    out = img.copy()
    out[..., 2] = np.roll(img[..., 2], 8, axis=1)
    out[..., 0] = np.roll(img[..., 0], -8, axis=1)
    h = out.shape[0]
    for _ in range(6):
        y, hh = np.random.randint(0, max(h - 24, 1)), np.random.randint(4, 20)
        out[y:y + hh] = np.roll(out[y:y + hh], np.random.randint(-30, 30), axis=1)
    return out


_VIG = {}


def f_vintage(img):
    k = np.array([[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]], np.float32)
    out = cv2.transform(img, k)
    h, w = img.shape[:2]
    if (h, w) not in _VIG:
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        _VIG[(h, w)] = np.clip(1.15 - 0.55 * r, 0.25, 1.0).astype(np.float32)[..., None]
    grain = np.random.normal(0, 9, (h, w, 1)).astype(np.float32)
    return np.clip(out.astype(np.float32) * _VIG[(h, w)] + grain, 0, 255).astype(np.uint8)


def f_negative(img):
    hsv = cv2.cvtColor(cv2.bitwise_not(img), cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + 60) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


STYLES = [
    ("Thermal / Heatmap", f_thermal), ("Neon Edges", f_neon), ("Pencil Sketch", f_sketch),
    ("Comic", f_comic), ("Pixel Art", f_pixel), ("Glitch", f_glitch),
    ("Vintage Film", f_vintage), ("Negative Dream", f_negative),
]


def painterly(img):
    """Slow, high-quality 'AI render' stand-in using OpenCV's NPR stylisation.

    Replace this function to call a real generative model (img2img API, local
    Stable Diffusion, ...). It must take and return a BGR uint8 image.
    """
    return cv2.detailEnhance(cv2.stylization(img, sigma_s=60, sigma_r=0.45), sigma_s=10, sigma_r=0.15)


AI_RENDER = painterly


def order_quad(pts):
    """Order 4 points as TL, TR, BR, BL (sorted by angle around their centre)."""
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)].astype(np.float32)


# ------------------------------------------------------------------- mode
class HandFrameMode:
    name = "HandFrame"

    def __init__(self):
        self.i = 0
        self.quad = None
        self.hist = deque(maxlen=8)   # recent quads, so a pinch can't drag the frame
        self.locked = None
        self.pinch_t0 = None
        self.fired = False
        self.ai = None
        self.ai_busy = False

    def on_key(self, k):
        if k in (ord("n"), ord("N")):
            self._step(1)
        elif k in (ord("p"), ord("P")):
            self._step(-1)
        elif k in (ord("c"), ord("C")):
            self.ai = None
        else:
            return False
        return True

    def _step(self, d):
        self.i = (self.i + d) % len(STYLES)
        self.ai = None

    def _start_ai(self, frame):
        if self.ai_busy:
            return
        h, w = frame.shape[:2]
        snap = cv2.resize(frame, (WORK_WIDTH, int(WORK_WIDTH * h / w))).copy()
        self.ai_busy = True

        def job():
            try:
                self.ai = AI_RENDER(snap)
            except Exception as e:  # keep the app alive whatever the renderer does
                print("AI render failed:", e)
            finally:
                self.ai_busy = False

        threading.Thread(target=job, daemon=True).start()

    # ----------------------------------------------------------- update
    def update(self, frame, hands, hud=True, ctx=None):
        h, w = frame.shape[:2]
        now = time.time()
        pinching = any(pinch_ratio(lm) < PINCH_ON for lm in hands)

        # quick pinch = next filter, long pinch = AI render
        if pinching and self.pinch_t0 is None:
            self.pinch_t0, self.fired = now, False
        if pinching and not self.fired and now - self.pinch_t0 >= HOLD_SECONDS:
            self.fired = True
            self._start_ai(frame)
        if not pinching and self.pinch_t0 is not None:
            if not self.fired and now - self.pinch_t0 > 0.05:
                self._step(1)
            self.pinch_t0 = None
        hold = 0.0 if self.pinch_t0 is None or self.fired else min((now - self.pinch_t0) / HOLD_SECONDS, 1.0)

        # frame corners from the thumb + index tips of both hands
        if pinching:
            if self.quad is not None:
                if self.locked is None:
                    self.locked = (self.hist[0] if self.hist else self.quad).copy()
                self.quad = self.locked
        elif len(hands) == 2:
            self.locked = None
            q = order_quad(np.array([hands[0][4], hands[0][8], hands[1][4], hands[1][8]], np.float32))
            self.quad = q if self.quad is None else self.quad * 0.5 + q * 0.5
            self.hist.append(self.quad.copy())
        else:
            self.quad, self.locked = None, None
            self.hist.clear()

        if self.quad is not None:
            small = cv2.resize(frame, (WORK_WIDTH, int(WORK_WIDTH * h / w)))
            src = self.ai if self.ai is not None else STYLES[self.i][1](small)
            self._compose(frame, src)
            for p in self.quad:
                cv2.circle(frame, (int(p[0]), int(p[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
            if hold > 0:
                text(frame, f"HOLD... {hold * 100:.0f}%", (int(self.quad[0][0]), int(self.quad[0][1]) - 14),
                     0.6, (0, 255, 255), 2)
            if self.ai_busy:
                text(frame, "AI RENDERING...", (int(self.quad[0][0]), int(self.quad[0][1]) - 14),
                     0.6, (0, 255, 255), 2)
        elif hud:
            text(frame, "Make a frame with both hands (thumb + index L-shapes)", (w // 2 - 270, h // 2), 0.7,
                 (0, 255, 255), 2)

        if hud:
            panel(frame, 0, TOP, w, 46, 0.55)
            label = "AI render (painterly)" if self.ai is not None else STYLES[self.i][0]
            text(frame, f"Style: {label}", (12, TOP + 20), 0.65, (255, 255, 255), 2)
            text(frame, "Quick pinch = next filter  |  hold pinch 0.6s = AI render  |  [N]/[P] style  [C] clear render",
                 (12, TOP + 38), 0.45, (0, 255, 255))

    def _compose(self, frame, src):
        """Warp `src` into the hand-made quad with a drop shadow and white border."""
        h, w = frame.shape[:2]
        q = self.quad
        x0, y0 = int(max(q[:, 0].min() - 40, 0)), int(max(q[:, 1].min() - 40, 0))
        x1, y1 = int(min(q[:, 0].max() + 40, w)), int(min(q[:, 1].max() + 40, h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        qr = (q - np.array([x0, y0], np.float32)).astype(np.float32)
        sh, sw = src.shape[:2]
        try:
            M = cv2.getPerspectiveTransform(np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]]), qr)
        except cv2.error:
            return
        roi = frame[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]
        warped = cv2.warpPerspective(src, M, (rw, rh), flags=cv2.INTER_LINEAR)
        mask = np.zeros((rh, rw), np.uint8)
        cv2.fillConvexPoly(mask, qr.astype(np.int32), 255)
        shadow = np.zeros((rh, rw), np.uint8)
        cv2.fillConvexPoly(shadow, (qr + np.array([8, 12], np.float32)).astype(np.int32), 255)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 10)
        a = (shadow.astype(np.float32) / 255 * 0.5)[..., None]
        roi[:] = (roi * (1 - a)).astype(np.uint8)
        roi[mask > 0] = warped[mask > 0]
        cv2.polylines(roi, [qr.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
