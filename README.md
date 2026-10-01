# 🖐️ HandLab

**Three hand-gesture computer-vision projects in one real-time webcam app.**

Built with Python, OpenCV and MediaPipe. Switch between modes with the number keys.

| Key | Mode | What it does |
|-----|------|--------------|
| `1` | **Invisibility** | Pinch with both hands to open a portal. Inside it, you disappear and the empty background shows through. |
| `2` | **HandFrame AI** | Frame your face with your hands. A live, stylised view floats inside the frame; pinch to change filters. |
| `3` | **Sign Language** | Recognises ASL fingerspelling and turns held signs into letters, words and sentences. |

<!-- Add a demo GIF here: ![demo](assets/demo.gif) -->

## 🚀 Quick start

```bash
git clone https://github.com/<your-username>/handlab.git
cd handlab
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Requirements: Python 3.10–3.12 and a webcam. Newer MediaPipe versions download a ~8 MB hand model on first run (internet needed once).

Options: `python main.py --camera 1 --mode 3 --width 1280 --height 720`

**Global keys:** `1` `2` `3` switch mode · `h` hide/show HUD · `f` fullscreen · `q` / `Esc` quit

---

## 1 · Invisibility

1. Press **`B`**, then step out of the frame for 3 seconds. The app captures the empty room (keep the camera still).
2. Come back and **pinch (thumb + index) with both hands**. The two pinch points are opposite corners of a yellow portal.
3. Inside the portal you vanish. Release the pinch and the portal closes.

How it works: pixels inside the portal that differ from the captured background are replaced by the background (with exposure compensation, hole filling and soft edges). `R` resets the background.

*Tips:* steady camera, steady lighting, and avoid clothing that exactly matches the wall.

## 2 · HandFrame AI

Make an "L" with the thumb and index finger of **each** hand to form a frame. A filtered live view is perspective-warped into your hand-made frame with a shadow and border.

- **Quick pinch** → next filter
- **Hold pinch 0.6 s** → "AI render": freezes a snapshot and paints it
- `N` / `P` next / previous style · `C` clear the render

Styles: Thermal/Heatmap, Neon Edges, Pencil Sketch, Comic, Pixel Art, Glitch, Vintage Film, Negative Dream.

> **About "AI render":** by default it uses OpenCV's painterly stylisation, which runs locally and offline. To use a real generative model, replace `painterly()` / `AI_RENDER` in `handframe.py` with a function that takes and returns a BGR image (for example an img2img API call).

## 3 · Sign Language (ASL fingerspelling)

Show a sign and **hold it ~0.9 s** to type it. Drop your hand between double letters.

| Key | Action |
|-----|--------|
| `Space` | finish the current word |
| `Backspace` | delete |
| `Enter` | clear everything |
| `Tab` | toggle **record mode** |

**Two recognisers:**

- **Starter rules** (default): hand-written geometry rules for `A B D F I L U V W Y`, plus an open palm = space. Approximate, so expect some misreads.
- **Your own signs** (recommended): press `Tab`, show a sign, press its letter key to record 30 samples. A k-nearest-neighbour classifier uses them immediately, with no training step, and saves them to `asl_samples.npz`. In record mode, `Space` records a SPACE gesture and `Backspace` records a DEL gesture. Once you have samples, they replace the starter rules.

Samples are normalised for position and scale, so they work at different distances from the camera. Record a few angles and both hands' worth if you can.

## 🧪 Tests

No camera needed. Tests use synthetic hands and frames.

```bash
python tests/test_logic.py
# optional: HANDLAB_PREVIEW=./preview python tests/test_logic.py   (saves images)
```

## 📁 Structure

```
handlab/
├── main.py           # app loop, mode switching
├── tracker.py        # MediaPipe hand tracking (old + new API)
├── invisibility.py   # mode 1
├── handframe.py      # mode 2
├── signlang.py       # mode 3
├── utils.py          # geometry + drawing helpers
├── tests/test_logic.py
└── requirements.txt
```

## 🛠️ Troubleshooting

- **Camera won't open:** try `--camera 1`, and close other apps using the webcam.
- **`pip install mediapipe` fails:** use Python 3.10–3.12.
- **Model download fails:** download `hand_landmarker.task` from the MediaPipe hand landmarker docs and place it next to `tracker.py`.
- **Portal shows ghosting:** re-capture the background with `B`; don't move the camera.
- **Signs misread:** switch to your own samples (`Tab`) and record each sign a few times.

## 💡 Ideas

Text-to-speech for the sentence, a full A–Z dataset, word auto-complete, recording the output to video, and a real diffusion backend for AI render.

## 📄 License

MIT. Add a `LICENSE` file to make it official.
