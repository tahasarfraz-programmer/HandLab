"""Mode 3 - Real-time sign language (ASL fingerspelling) -> letters -> words -> sentence.

Two recognisers:
  * Starter rules   - a small hand-written rule set (A B D F I L U V W Y + open palm = SPACE).
                      Works out of the box but is only approximate.
  * Your own model  - press TAB, show a sign, press its letter key to record samples.
                      A k-nearest-neighbour classifier uses them instantly (no training step).
                      Once you have samples, they replace the starter rules.

Hold a sign for ~0.9 s to type it. Drop your hand between double letters.
"""
import os
import time
from collections import Counter, deque

import cv2
import numpy as np

from utils import (TOP, bar, dist, draw_skeleton, hand_bbox, palm_size, panel,
                   pinch_ratio, text)

SAMPLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asl_samples.npz")
HOLD_SECONDS = 0.9
RECORD_COUNT = 30
KNN_K = 5
KNN_MAX_DIST = 0.9   # reject predictions whose neighbours are farther than this


# --------------------------------------------------------------- features
def features(lm):
    """Translation- and scale-invariant 42-vector from 21 (x, y) landmarks."""
    a = np.array(lm, np.float32)
    a -= a[0]
    a /= np.linalg.norm(a, axis=1).max() + 1e-6
    return a.flatten()


class KNN:
    def __init__(self):
        self.X = np.zeros((0, 42), np.float32)
        self.y = []

    def load(self, path):
        if os.path.exists(path):
            d = np.load(path, allow_pickle=False)
            self.X, self.y = d["X"].astype(np.float32), d["y"].tolist()

    def save(self, path):
        np.savez(path, X=self.X, y=np.array(self.y))

    def add(self, f, label):
        self.X = np.vstack([self.X, f[None]])
        self.y.append(label)

    def predict(self, f):
        if len(self.y) == 0:
            return None, 0.0
        d = np.linalg.norm(self.X - f, axis=1)
        idx = np.argsort(d)[:KNN_K]
        if d[idx].mean() > KNN_MAX_DIST:
            return None, 0.0
        votes = {}
        for i in idx:
            votes[self.y[i]] = votes.get(self.y[i], 0.0) + 1.0 / (d[i] + 0.05)
        best = max(votes, key=votes.get)
        return best, votes[best] / sum(votes.values())


# ------------------------------------------------------------ starter rules
def finger_states(lm):
    """(thumb_out, [index, middle, ring, pinky extended?]) from landmark geometry."""
    ps = palm_size(lm)
    ext = [dist(lm[t], lm[0]) > dist(lm[p], lm[0]) * 1.12 for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]
    thumb = dist(lm[4], lm[17]) / ps > 1.05
    return thumb, ext


def rule_predict(lm):
    thumb, (i, m, r, p) = finger_states(lm)
    if pinch_ratio(lm) < 0.30 and m and r and p:
        return "F", 0.8
    if thumb and i and m and r and p:
        return "SPACE", 0.8
    if thumb and i and not (m or r or p):
        return "L", 0.8
    if thumb and p and not (i or m or r):
        return "Y", 0.8
    if not thumb:
        if i and m and r and p:
            return "B", 0.8
        if i and m and r:
            return "W", 0.8
        if i and m and not (r or p):
            return ("V" if dist(lm[8], lm[12]) / palm_size(lm) > 0.55 else "U"), 0.8
        if i and not (m or r or p):
            return "D", 0.8
        if p and not (i or m or r):
            return "I", 0.8
        if not (i or m or r or p):
            return "A", 0.8
    return None, 0.0


# ------------------------------------------------------------------- mode
class SignMode:
    name = "Sign Language"

    def __init__(self, path=SAMPLES_PATH):
        self.path = path
        self.knn = KNN()
        self.knn.load(path)
        self.word, self.sentence = "", []
        self.recent = deque(maxlen=7)
        self.cand, self.cand_t, self.armed = None, 0.0, True
        self.record, self.rec_label, self.rec_left = False, None, 0

    # -------------------------------------------------------- text edit
    def commit_word(self):
        if self.word:
            self.sentence.append(self.word)
            self.word = ""

    def delete(self):
        if self.word:
            self.word = self.word[:-1]
        elif self.sentence:
            self.sentence[-1] = self.sentence[-1][:-1]
            if not self.sentence[-1]:
                self.sentence.pop()

    def _commit(self, label):
        if label == "SPACE":
            self.commit_word()
        elif label == "DEL":
            self.delete()
        else:
            self.word += label

    # ------------------------------------------------------------ input
    def on_key(self, k):
        if k == 9:                                   # TAB
            self.record, self.rec_left = not self.record, 0
            return True
        if self.record:
            if 65 <= k <= 90 or 97 <= k <= 122:
                self.rec_label, self.rec_left = chr(k).upper(), RECORD_COUNT
                return True
            if k == 32:
                self.rec_label, self.rec_left = "SPACE", RECORD_COUNT
                return True
            if k in (8, 127):
                self.rec_label, self.rec_left = "DEL", RECORD_COUNT
                return True
            return False
        if k == 32:
            self.commit_word()
        elif k in (8, 127):
            self.delete()
        elif k == 13:
            self.word, self.sentence = "", []
        else:
            return False
        return True

    # ----------------------------------------------------------- update
    def update(self, frame, hands, hud=True, ctx=None):
        ctx = ctx or {"fps": 0, "ms": 0}
        now = time.time()
        label, conf = None, 0.0
        have_model = len(self.knn.y) >= KNN_K

        if hands:
            lm = hands[0]
            if self.rec_left > 0 and self.rec_label:
                self.knn.add(features(lm), self.rec_label)
                self.rec_left -= 1
                if self.rec_left == 0:
                    self.knn.save(self.path)
            label, conf = self.knn.predict(features(lm)) if have_model else rule_predict(lm)

        self.recent.append(label)
        top = Counter(self.recent).most_common(1)[0]
        stable = top[0] if top[0] is not None and top[1] >= 5 else None

        if self.record:
            self.cand, self.armed = None, True
        elif stable is None:
            self.cand, self.armed = None, True
        elif stable != self.cand:
            self.cand, self.cand_t, self.armed = stable, now, True
        elif self.armed and now - self.cand_t >= HOLD_SECONDS:
            self._commit(stable)
            self.armed = False
        progress = 0.0 if (self.cand is None or not self.armed) else min((now - self.cand_t) / HOLD_SECONDS, 1.0)

        for lm in hands:
            draw_skeleton(frame, lm, (0, 190, 255), (0, 0, 255), 2)
        if hands:
            x0, y0, x1, y1 = hand_bbox(hands[0], 25, frame.shape)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 230, 0), 2)
            tag = f"Sign: {label} {conf * 100:.0f}%" if label else "Sign: ?"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x0, y0 - th - 10), (x0 + tw + 10, y0), (0, 230, 0), -1)
            cv2.putText(frame, tag, (x0 + 5, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        if hud:
            self._hud(frame, hands, label, conf, progress, have_model, ctx)

    # -------------------------------------------------------------- hud
    def _hud(self, frame, hands, label, conf, progress, have_model, ctx):
        h, w = frame.shape[:2]
        mode = (f"RECORDING '{self.rec_label}' {RECORD_COUNT - self.rec_left}/{RECORD_COUNT}"
                if self.rec_left > 0 else ("RECORD MODE" if self.record else "Tracking"))
        model = f"your samples ({len(self.knn.y)})" if have_model else "starter rules"

        panel(frame, 10, TOP + 6, 340, 108)
        text(frame, "SIGN LANGUAGE RECOGNITION", (20, TOP + 26), 0.5, (255, 200, 0))
        text(frame, f"FPS: {ctx['fps']:.0f}    Latency: {ctx['ms']:.0f} ms", (20, TOP + 46), 0.45)
        text(frame, f"Mode: {mode}    Hands: {len(hands)}", (20, TOP + 66), 0.45, (0, 255, 255))
        text(frame, f"Model: {model}", (20, TOP + 86), 0.45)
        text(frame, "[Tab] record mode", (20, TOP + 104), 0.4, (170, 170, 170))

        px = w - 300
        panel(frame, px, TOP + 6, 290, 130)
        text(frame, "RECOGNITION", (px + 10, TOP + 26), 0.5, (255, 200, 0))
        text(frame, f"Current: {label or '-'}", (px + 10, TOP + 50), 0.7, (255, 255, 255), 2)
        bar(frame, px + 10, TOP + 58, 270, 10, conf)
        text(frame, "Hold to type", (px + 10, TOP + 86), 0.42, (200, 200, 200))
        bar(frame, px + 10, TOP + 92, 270, 10, progress, (0, 200, 255))
        text(frame, f"Word buffer: {self.word}", (px + 10, TOP + 124), 0.55, (0, 255, 255), 1)

        sentence = " ".join(self.sentence + ([self.word] if self.word else []))
        sentence = sentence[:1].upper() + sentence[1:].lower()
        panel(frame, 0, h - 74, w, 74, 0.7)
        text(frame, sentence or "...", (16, h - 38), 0.9, (255, 255, 255), 2)
        text(frame, "[Space] add word   [Backspace] delete   [Enter] clear", (16, h - 12), 0.45, (0, 255, 255), 1,
             shadow=False)

        if self.record:
            panel(frame, w // 2 - 330, TOP + 150, 660, 78, 0.75)
            text(frame, "RECORD MODE: show a sign, then press its letter key (30 samples).", (w // 2 - 315, TOP + 176),
                 0.55, (0, 255, 255), 1)
            text(frame, "Space key = SPACE gesture   Backspace key = DEL gesture   Tab = exit", (w // 2 - 315, TOP + 198),
                 0.5)
            counts = Counter(self.knn.y)
            text(frame, "Samples: " + (" ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "none"),
                 (w // 2 - 315, TOP + 218), 0.45, (170, 255, 170))
