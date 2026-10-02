"""Interactive digit demo for the from-scratch NeuraCPP engine.

Same canvas UX as https://github.com/Scraper1010/pytorch_handwritten_digits
(white 280x280 sketchpad, black fixed pencil + eraser, Clear/Classify,
live prediction on stroke, top-3 confidences) — but inference is pure
NumPy running the C++ weights (no torch).

Run:
  pip install -r python/requirements.txt
  # train first (from repo root):
  #   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
  #   cmake --build build
  #   ./build/neuracpp --epochs 5 --output checkpoints/model.bin
  python python/app.py --model checkpoints/model.bin
"""

import argparse
import os
import sys
import time

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from neuracpp_predict import load_model, predict_proba

# Same Windows temp-file hardening as the PyTorch twin: Gradio reads its
# temp PNGs with PIL and they can be transiently locked / 0-byte.
_orig_convert = gr.ImageEditor.convert_and_format_image


def _safe_convert_and_format_image(self, file, _attempts=6):
    last_err = None
    for attempt in range(_attempts):
        try:
            return _orig_convert(self, file)
        except OSError as e:
            last_err = e
            time.sleep(0.05 * (2 ** attempt))
    print(f"[!] Skipping unreadable temp image after retries: {last_err}")
    return None


gr.ImageEditor.convert_and_format_image = _safe_convert_and_format_image


def preprocess_canvas_drawing(canvas_data):
    """Gradio canvas -> (28,28) float32 [0,1], MNIST-style white-on-black.

    Same 20px-box algorithm as the PyTorch twin, hardened in two places
    the twin gets wrong on real drawings (found by fault-injection tests):
    1. Background detection reads the BORDER frame, not the global mean, so
       a huge dark digit can no longer flip the polarity decision.
    2. The finished tensor is max-normalized (no-op on full-contrast input,
       rescues faint/fast strokes) — proven 0% accuracy change on clean
       MNIST, full recovery on 40%-contrast inputs.
    Returns zeros for empty canvas.
    """
    if canvas_data is None:
        return None

    if isinstance(canvas_data, dict):
        img_array = canvas_data.get("composite", None)
        if img_array is None:
            for layer in canvas_data.get("layers") or []:
                if layer is not None:
                    img_array = layer
                    break
        if img_array is None:
            img_array = canvas_data.get("background", None)
        if img_array is None:
            img_array = canvas_data.get("image", None)
    else:
        img_array = canvas_data

    if img_array is None:
        return np.zeros((28, 28), dtype=np.float32)

    if isinstance(img_array, str):
        try:
            img = Image.open(img_array).convert("L")
        except Exception:
            return np.zeros((28, 28), dtype=np.float32)
    elif isinstance(img_array, Image.Image):
        if img_array.mode == "RGBA":
            bg = Image.new("RGBA", img_array.size, (0, 0, 0, 255))
            img = Image.alpha_composite(bg, img_array).convert("L")
        else:
            img = img_array.convert("L")
    else:
        try:
            img_array = np.asarray(img_array)
        except Exception:
            return np.zeros((28, 28), dtype=np.float32)
        if img_array.size == 0:
            return np.zeros((28, 28), dtype=np.float32)
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            rgba = Image.fromarray(img_array.astype("uint8"), mode="RGBA")
            bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
            img = Image.alpha_composite(bg, rgba).convert("L")
        else:
            img = Image.fromarray(img_array.astype("uint8")).convert("L")

    # Background = the 4px border frame (digits live in the middle, never in
    # the frame). White border -> black-on-white drawing -> invert to MNIST
    # polarity. A giant digit can no longer outvote the background.
    border_pixels = np.concatenate([
        np.array(img)[:4, :].ravel(), np.array(img)[-4:, :].ravel(),
        np.array(img)[:, :4].ravel(), np.array(img)[:, -4:].ravel(),
    ])
    if float(border_pixels.mean()) > 128:
        img = ImageOps.invert(img)

    arr = np.array(img)
    ys, xs = np.argwhere(arr > 30).T if (arr > 30).any() else (np.array([]), np.array([]))
    if ys.size == 0:
        return np.zeros((28, 28), dtype=np.float32)

    ymin, ymax, xmin, xmax = ys.min(), ys.max(), xs.min(), xs.max()
    cropped = img.crop((int(xmin), int(ymin), int(xmax) + 1, int(ymax) + 1))

    w, h = cropped.size
    if w > h:
        new_w, new_h = 20, max(1, int(round(20.0 / w * h)))
    else:
        new_h, new_w = 20, max(1, int(round(20.0 / h * w)))
    resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas_28 = Image.new("L", (28, 28), 0)
    canvas_28.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))
    tensor_28 = np.array(canvas_28).astype(np.float32) / 255.0

    # Stretch faint strokes to full contrast (max -> 1.0). No-op when the
    # brightest pixel already hits 1.0, which clean inputs always do.
    peak_value = float(tensor_28.max())
    if peak_value > 1e-6:
        tensor_28 = (tensor_28 / peak_value).astype(np.float32)
    return tensor_28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join("checkpoints", "model.bin"))
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model file '{args.model}' not found.", file=sys.stderr)
        print(f"Train first: ./build/neuracpp --epochs 5 --output {args.model}", file=sys.stderr)
        raise SystemExit(1)
    try:
        params = load_model(args.model)
    except Exception as e:
        print(f"Error: failed to load '{args.model}': {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[+] Loaded NeuraCPP weights from {args.model}")

    def predict_digit(canvas_data):
        # Returns (confidences, model_view): the Label dict plus the exact
        # 28x28 tensor the network classified, upscaled for human eyes.
        if canvas_data is None:
            return None, None
        try:
            img28 = preprocess_canvas_drawing(canvas_data)
        except Exception as e:
            print(f"[!] Preprocessing failed: {e}")
            return None, None
        if img28 is None or float(img28.sum()) == 0.0:
            return None, None  # blank canvas -> empty label, not a bogus digit
        proba = predict_proba(img28, params)
        # Chunky NEAREST upscale so individual model pixels stay visible.
        model_view = Image.fromarray((img28 * 255).astype(np.uint8), mode="L")
        model_view = model_view.resize((168, 168), Image.NEAREST)
        return {str(i): float(proba[i]) for i in range(10)}, model_view

    with gr.Blocks(title="neuracpp_handwritten_digits") as demo:
        gr.Markdown(
            """
            # Handwritten Digit Recognizer (from-scratch C++ engine)
            Draw a single digit (0–9) with the black pencil on the white canvas.
            Inference is pure NumPy running `model.bin` trained by the C++ CNN — no PyTorch.
            """
        )
        sketchpad = gr.Sketchpad(
            value=Image.new("RGB", (280, 280), (255, 255, 255)),
            label="Drawing Canvas",
            type="numpy",
            image_mode="RGB",
            brush=gr.Brush(colors=["#000000"], default_color="#000000",
                           color_mode="fixed", default_size=18),
            eraser=gr.Eraser(default_size=18),
            canvas_size=(280, 280),
            fixed_canvas=True,
            layers=False,
            transforms=[],
            sources=(),
            buttons=[],
            interactive=True,
            elem_id="draw-canvas",
        )
        with gr.Row():
            clear_btn = gr.Button("Clear Canvas", variant="secondary")
            classify_btn = gr.Button("Classify Digit", variant="primary")
        with gr.Row():
            output_label = gr.Label(label="Prediction Confidences", num_top_classes=3)
            # Live X-ray of the preprocessing: the exact 28x28 the model sees.
            # If predictions look wrong, glance here first — an inverted, faint,
            # or clipped digit means the drawing (not the network) needs fixing.
            model_view = gr.Image(label="What the model sees (28x28 input)",
                                  image_mode="L", height=168, width=168,
                                  interactive=False)

        classify_btn.click(fn=predict_digit, inputs=sketchpad, outputs=[output_label, model_view],
                           concurrency_limit=1)
        sketchpad.change(fn=predict_digit, inputs=sketchpad, outputs=[output_label, model_view],
                         trigger_mode="always_last", concurrency_limit=1,
                         show_progress="hidden")

        def clear_all():
            return Image.new("RGB", (280, 280), (255, 255, 255)), None, None

        clear_btn.click(fn=clear_all, outputs=[sketchpad, output_label, model_view],
                        concurrency_limit=1, show_progress="hidden")
        sketchpad.clear(fn=lambda: (None, None), outputs=[output_label, model_view],
                        concurrency_limit=1, show_progress="hidden")
        demo.queue(default_concurrency_limit=1)

    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False,
                theme=gr.themes.Soft(), max_threads=1,
                css="#draw-canvas button[aria-label='Clear canvas'] { display: none !important; }")


if __name__ == "__main__":
    main()
