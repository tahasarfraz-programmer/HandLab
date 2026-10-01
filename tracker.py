"""MediaPipe hand tracking that works with both old (`mp.solutions`) and new (Tasks) APIs."""
import os
import time
import urllib.request

import cv2

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


class HandTracker:
    def __init__(self, max_hands=2):
        import mediapipe as mp
        self.mp = mp
        self.legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        if self.legacy:
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=max_hands, model_complexity=1,
                min_detection_confidence=0.6, min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks.python import BaseOptions, vision
            self._ensure_model()
            opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=0.6,
                min_tracking_confidence=0.5)
            self.hands = vision.HandLandmarker.create_from_options(opts)
        self._t0 = time.monotonic()
        self._last_ts = -1

    @staticmethod
    def _ensure_model():
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
            return
        print("Downloading hand model (one time, ~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    def process(self, frame_bgr):
        """Return a list of hands; each hand is 21 (x, y) pixel tuples."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = []
        if self.legacy:
            res = self.hands.process(rgb)
            for hl in (res.multi_hand_landmarks or []):
                out.append([(p.x * w, p.y * h) for p in hl.landmark])
        else:
            ts = max(int((time.monotonic() - self._t0) * 1000), self._last_ts + 1)
            self._last_ts = ts
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            res = self.hands.detect_for_video(img, ts)
            for hl in res.hand_landmarks:
                out.append([(p.x * w, p.y * h) for p in hl])
        return out
