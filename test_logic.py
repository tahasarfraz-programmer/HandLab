"""Camera-free tests: synthetic hands + synthetic frames exercise every mode.

Run:  python tests/test_logic.py
Optional env vars:
  HANDLAB_PREVIEW=<dir>      save preview images there
  HANDLAB_TEST_IMAGE=<file>  use this photo instead of a synthetic scene
"""
import math
import os
import sys
import tempfile
import time as _time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import handframe  # noqa: E402
import invisibility  # noqa: E402
import signlang  # noqa: E402
from utils import palm_size, pinch_ratio  # noqa: E402

PREVIEW = os.environ.get("HANDLAB_PREVIEW")
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- fake hands
MCP = [(-30, -90), (-8, -95), (14, -90), (35, -80)]
# label: (thumb_out, [index, middle, ring, pinky] extended, spread, pinch)
POSES = {
    "A": (False, [0, 0, 0, 0], False, False), "B": (False, [1, 1, 1, 1], False, False),
    "D": (False, [1, 0, 0, 0], False, False), "F": (False, [0, 1, 1, 1], False, True),
    "I": (False, [0, 0, 0, 1], False, False), "L": (True, [1, 0, 0, 0], False, False),
    "U": (False, [1, 1, 0, 0], False, False), "V": (False, [1, 1, 0, 0], True, False),
    "W": (False, [1, 1, 1, 0], False, False), "Y": (True, [0, 0, 0, 1], False, False),
    "SPACE": (True, [1, 1, 1, 1], False, False),
}


def make_hand(label, cx=600, cy=500, s=1.0, rot=0.0, noise=0.0):
    thumb_out, ext, spread, pinch = POSES[label]
    P = [(0, 0)]
    P += ([(-20, -15), (-40, -35), (-65, -50), (-90, -60)] if thumb_out
          else [(-20, -15), (-30, -35), (-18, -50), (0, -50)])
    dirs = [(-0.05, -1), (0, -1), (0.05, -1), (0.12, -1)]
    if spread:
        dirs[0], dirs[1] = (-0.35, -0.94), (0.35, -0.94)
    for j in range(4):
        mx, my = MCP[j]
        dx, dy = dirs[j]
        n = math.hypot(dx, dy)
        dx, dy = dx / n, dy / n
        offs = (0, 45, 70, 90) if ext[j] else (0, 35, 8, -10)
        P += [(mx + dx * o, my + dy * o) for o in offs]
    if pinch:
        P[4] = (P[8][0] + 3, P[8][1] + 3)
    P = np.array(P, np.float32)
    c, sn = math.cos(rot), math.sin(rot)
    P = P @ np.array([[c, -sn], [sn, c]], np.float32).T * s
    P += rng.normal(0, noise, P.shape) + np.array([cx, cy])
    return [tuple(p) for p in P]


def save(name, img):
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        cv2.imwrite(os.path.join(PREVIEW, name), img)


def scene(w=1280, h=720):
    path = os.environ.get("HANDLAB_TEST_IMAGE")
    if path:
        return cv2.resize(cv2.imread(path), (w, h))
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 255 // w), (yy * 255 // h), 255 - (xx * 255 // w)], -1).astype(np.uint8)
    cv2.circle(img, (w // 2, h // 2), 120, (200, 220, 240), -1)     # a "face"
    return img


class FakeTime:
    def __init__(self):
        self.t = 1000.0

    def time(self):
        return self.t


# --------------------------------------------------------------------- tests
def test_rules():
    for label in POSES:
        for rot in (0, 0.2, -0.2):
            got, _ = signlang.rule_predict(make_hand(label, rot=rot, s=rng.uniform(0.7, 1.4)))
            assert got == label, f"rules: {label} (rot {rot}) -> {got}"
    print("ok  starter rules recognise", " ".join(POSES))


def test_knn():
    knn = signlang.KNN()
    for label in POSES:
        for _ in range(30):
            knn.add(signlang.features(make_hand(label, rng.integers(300, 900), rng.integers(300, 600),
                                                rng.uniform(0.7, 1.4), rng.uniform(-0.15, 0.15), 1.5)), label)
    path = os.path.join(tempfile.mkdtemp(), "s.npz")
    knn.save(path)
    knn2 = signlang.KNN()
    knn2.load(path)
    assert knn2.y == knn.y and np.allclose(knn2.X, knn.X)
    ok = n = 0
    for label in POSES:
        for _ in range(40):
            got, _ = knn2.predict(signlang.features(make_hand(label, rng.integers(300, 900), rng.integers(300, 600),
                                                              rng.uniform(0.7, 1.4), rng.uniform(-0.15, 0.15), 1.5)))
            ok += got == label
            n += 1
    acc = ok / n
    assert acc > 0.95, f"knn accuracy {acc:.2f}"
    junk = signlang.KNN()
    junk.X, junk.y = knn.X, knn.y
    r, _ = junk.predict(rng.normal(0, 1, 42).astype(np.float32))
    assert r is None, "random garbage should be rejected"
    print(f"ok  kNN accuracy {acc * 100:.1f}% on unseen noisy/rotated/scaled hands; save/load; garbage rejected")


def test_sentence():
    ft = FakeTime()
    signlang.time = ft
    mode = signlang.SignMode(path=os.path.join(tempfile.mkdtemp(), "none.npz"))
    frame = np.zeros((720, 1280, 3), np.uint8)

    def show(label, secs):
        for _ in range(int(secs * 30)):
            ft.t += 1 / 30
            mode.update(frame, [make_hand(label)] if label else [], True, {"fps": 30, "ms": 5})

    for lab in ["B", None, "A", None, "D", None, "SPACE", None, "Y", None, "I", None]:
        show(lab, 1.2 if lab else 0.3)
    assert mode.sentence == ["BAD"] and mode.word == "YI", (mode.sentence, mode.word)
    mode.on_key(8)
    assert mode.word == "Y"
    mode.on_key(32)
    assert mode.sentence == ["BAD", "Y"]
    save("sign_hud.png", frame)
    signlang.time = _time
    print("ok  hold-to-type spells BAD + space + YI, backspace/space keys work")


def test_record_mode():
    ft = FakeTime()
    signlang.time = ft
    path = os.path.join(tempfile.mkdtemp(), "rec.npz")
    mode = signlang.SignMode(path=path)
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert mode.on_key(9) and mode.record
    assert mode.on_key(ord("q")) and mode.rec_label == "Q"      # letters are captured, not "quit"
    for _ in range(40):
        ft.t += 1 / 30
        mode.update(frame, [make_hand("A")], True, None)
    assert mode.knn.y.count("Q") == signlang.RECORD_COUNT and os.path.exists(path)
    save("sign_record.png", frame)
    signlang.time = _time
    print("ok  record mode captures 30 samples for 'Q' and saves them")


def test_invisibility():
    h, w = 720, 1280
    yy, xx = np.mgrid[0:h, 0:w]
    plate = (np.stack([128 + 60 * np.sin(xx / 70), 128 + 60 * np.cos(yy / 50), 140 + 50 * np.sin((xx + yy) / 90)], -1)
             ).astype(np.uint8)
    frame0 = (plate * 0.93).astype(np.uint8)                     # slight exposure drift
    cv2.ellipse(frame0, (640, 380), (150, 260), 0, 0, 360, (40, 40, 180), -1)   # "you"
    m = invisibility.InvisibilityMode()
    m.plate = plate
    hand_a, hand_b = make_hand("A", 300, 600), make_hand("A", 900, 600)
    for hd, pt in ((hand_a, (420, 200)), (hand_b, (860, 560))):    # pinch at opposite corners
        hd[4], hd[8] = pt, (pt[0] + 4, pt[1] + 4)
    out = None
    for _ in range(40):
        out = frame0.copy()
        m.update(out, [hand_a, hand_b], True, None)
    assert m.progress > 0.95, m.progress
    ref = (plate * 0.93).astype(np.float32)
    # portal spans x 420..860, y 200..560; the "person" ellipse spans y 120..640
    inside = np.abs(out[260:520, 560:720].astype(np.float32) - ref[260:520, 560:720]).mean()
    assert inside < 12, f"person not removed inside portal ({inside:.1f})"
    below = out[600, 640].astype(int)                      # below the portal, still on the person
    assert np.abs(below - np.array([40, 40, 180])).max() < 25, f"person should remain outside portal: {below}"
    save("invisibility_before.png", frame0)
    save("invisibility_after.png", out)
    m.update(out, [], True, None)
    print(f"ok  portal opens (progress {m.progress:.2f}); person gone inside (err {inside:.1f}); outside kept")


def test_handframe():
    ft = FakeTime()
    handframe.time = ft
    base = scene()
    m = handframe.HandFrameMode()
    left, right = make_hand("A", 300, 650), make_hand("A", 980, 650)
    left[8], left[4] = (330, 150), (330, 480)
    right[8], right[4] = (950, 150), (950, 480)
    frame = base.copy()
    for _ in range(10):
        frame = base.copy()
        m.update(frame, [left, right], True, None)
    assert m.quad is not None and np.abs(frame.astype(int) - base.astype(int)).sum() > 0
    save("handframe_0.png", frame)

    # every style runs and keeps shape/dtype
    small = cv2.resize(base, (640, 360))
    tiles = []
    for name, fn in handframe.STYLES:
        out = fn(small)
        assert out.shape == small.shape and out.dtype == np.uint8, name
        tiles.append(cv2.putText(cv2.resize(out, (320, 180)), name, (6, 20), 0, 0.55, (255, 255, 255), 2))
    save("handframe_styles.png", np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))

    # quick pinch -> next filter
    pinch = make_hand("A", 300, 650)
    pinch[4], pinch[8] = (330, 480), (333, 483)
    assert pinch_ratio(pinch) < 0.3
    i0 = m.i
    for _ in range(6):
        ft.t += 0.03
        m.update(base.copy(), [pinch, right], True, None)
    for _ in range(3):
        ft.t += 0.03
        m.update(base.copy(), [left, right], True, None)
    assert m.i == (i0 + 1) % len(handframe.STYLES), "quick pinch should advance the style"

    # long pinch -> AI render
    for _ in range(30):
        ft.t += 0.03
        m.update(base.copy(), [pinch, right], True, None)
    for _ in range(100):
        if m.ai is not None:
            break
        _time.sleep(0.1)
    assert m.ai is not None, "hold-pinch should produce an AI render"
    for _ in range(3):
        ft.t += 0.03
        m.update(base.copy(), [left, right], True, None)
    i1 = m.i
    assert i1 == (i0 + 1) % len(handframe.STYLES), "long pinch must not also change the style"
    frame = base.copy()
    m.update(frame, [left, right], True, None)
    save("handframe_ai.png", frame)
    handframe.time = _time
    print("ok  frame warps into hand corners; 8 styles run; quick pinch=next, hold pinch=AI render (no double trigger)")


def test_quad_order():
    pts = np.array([[900, 480], [300, 150], [900, 150], [300, 480]], np.float32)
    q = handframe.order_quad(pts)
    assert q.tolist() == [[300, 150], [900, 150], [900, 480], [300, 480]], q.tolist()
    print("ok  quad corners ordered TL TR BR BL")


if __name__ == "__main__":
    test_quad_order()
    test_rules()
    test_knn()
    test_sentence()
    test_record_mode()
    test_invisibility()
    test_handframe()
    print("\nAll tests passed.")
