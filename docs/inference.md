# Python inference (`python/`)

Pure-NumPy twin of the C++ forward pass — no torch. Proves the math is
understood, not just the API.

## `neuracpp_predict.py`

Function-for-function mirror of the headers:

| Python | C++ equivalent |
|---|---|
| `load_model(path)` | `load_model(path, ...)` — same tag/shape/blob layout |
| `conv2d_forward` | `Conv2D::forward` (explicit pad + per-position tensordot) |
| `relu` / `maxpool2d` | `ReLU::forward` / `MaxPool2D::forward` |
| `softmax` | softmax half of `CrossEntropyLoss::forward` |
| `predict_proba` | `DigitNetwork::forward` + softmax |

```python
params = load_model("checkpoints/model.bin")
proba = predict_proba(canvas_28x28, params)  # (10,) white-on-black [0,1]
digit, confidences = predict_digit(canvas_28x28, params)
```

## `app.py` (Gradio demo)

Same canvas UX as the PyTorch twin: white 280×280 sketchpad, fixed black
pencil + eraser, Clear/Classify buttons, live prediction on stroke, top-3
confidences — plus a **"What the model sees"** panel showing the exact 28×28
input upscaled. Canvas → 28×28 preprocessing:

1. Grayscale (RGBA composited onto black),
2. Background read from the 4px **border frame** (not the global mean, which a
   giant dark digit can outvote) — white border inverts to MNIST polarity,
3. Bounding-box crop (pixels > 30), longest side fit to 20px,
4. Center-paste on black 28×28, divide by 255, then max-normalize (no-op on
   full-contrast input; proven 0% accuracy change on clean MNIST).

Blank canvas yields an empty label (not a bogus digit). **Missing checkpoint
is a hard error** — the app exits instead of launching without weights:

```
pip install -r python/requirements.txt
python python/app.py --model checkpoints/model.bin  # → http://127.0.0.1:7860
```
