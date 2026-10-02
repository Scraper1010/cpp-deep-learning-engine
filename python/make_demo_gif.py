"""Builds digit-writing GIFs by drawing in the app itself.

Like the PyTorch twin's scripts/make_demo_gif.py: each digit is a scripted
pen path (polyline waypoints on a 280px white canvas, 18px brush — the exact
app settings). Every animation frame draws the partial path, runs it through
the REAL app preprocessing (app.preprocess_canvas_drawing) and the trained
NumPy model, and captions the frame with the live prediction. No MNIST test
files involved — what you see is what the model outputs on genuine canvas
input, including a properly handwritten 5.

Outputs (all looping forever):
  assets/demo_grid.gif — all nine digits side-by-side in ONE file, so every
                         cell advances in perfect sync
  assets/demo.gif      — all nine digits back-to-back
  assets/demo_1.gif ... assets/demo_9.gif — one portrait per digit

Usage:
  python python/make_demo_gif.py
  python python/make_demo_gif.py --output assets/demo.gif --scale 5
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import preprocess_canvas_drawing
from neuracpp_predict import load_model, predict_proba

CANVAS_SIZE = 280      # Same canvas as the Gradio Sketchpad.
PEN_WIDTH = 18         # Same brush as the app (fixed black pencil).
REVEAL_FRAMES = 16     # Pen-sweep frames per digit.
HOLD_FRAMES = 4        # Verdict hold frames per digit.


def circle_stroke(center_x, center_y, radius, point_count=20, start_deg=-90):
    """Generates circle waypoints (for 8's loops), starting at start_deg."""
    waypoints = []
    for step in range(point_count + 1):
        angle = math.radians(start_deg + 360.0 * step / point_count)
        waypoints.append((center_x + radius * math.cos(angle),
                          center_y + radius * math.sin(angle)))
    return waypoints


def digit_strokes():
    """Hand-authored pen paths for digits 1-9 on the 280px canvas.

    Each digit is a list of strokes; each stroke is a waypoint polyline.
    Strokes are drawn in order with the app's 18px round brush.
    """
    return {
        # Flag top, straight stem, baseline foot.
        1: [[(105, 62), (138, 42), (138, 235)],
            [(100, 235), (178, 235)]],
        # Top arc, diagonal, baseline (continuous corners).
        2: [[(70, 70), (120, 45), (180, 50), (210, 80), (205, 120)],
            [(205, 120), (150, 170), (100, 210), (70, 225)],
            [(70, 225), (130, 225), (210, 225)]],
        # Upper bowl flowing into the lower bowl.
        3: [[(92, 62), (150, 44), (198, 68), (186, 112), (132, 126)],
            [(132, 126), (186, 140), (200, 184), (152, 220), (92, 212)]],
        # Open 4: diagonal down-left, crossbar, then the tall stem.
        4: [[(172, 52), (96, 166)],
            [(66, 166), (206, 166)],
            [(172, 52), (172, 236)]],
        # Flat top bar, then stem curving into the belly (a real 5).
        5: [[(92, 56), (196, 56)],
            [(96, 56), (90, 128), (135, 132), (182, 148),
             (186, 192), (148, 224), (92, 218)]],
        # Curved hook flowing into a tight round loop.
        6: [[(160, 60), (135, 78), (115, 115), (112, 165),
             (130, 200), (165, 210), (190, 185), (178, 155),
             (145, 148), (122, 162)]],
        # Slanted bar into a sharp corner, then the long diagonal.
        # (A flat bar reads as 2's top arc, a short bar reads as 1.)
        7: [[(72, 68), (198, 50)],
            [(198, 50), (112, 238)]],
        # Upper loop, then the lower loop (pen lifts between circles).
        8: [circle_stroke(140, 105, 44),
            circle_stroke(140, 190, 50)],
        # Top loop, then the tail down the right side.
        9: [circle_stroke(140, 104, 50),
            [(182, 116), (176, 184), (150, 234)]],
    }


def stroke_lengths(strokes):
    """Returns the drawn length of each stroke (for frame budgeting)."""
    lengths = []
    for stroke in strokes:
        stroke_length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(stroke[:-1], stroke[1:]))
        lengths.append(max(stroke_length, 1e-6))
    return lengths


def draw_partial(strokes, progress):
    """Renders strokes up to progress (0..1 of total pen length) on canvas.

    Finished strokes draw whole; the in-progress stroke draws up to the
    interpolated pen tip. Each stroke is its own line() call, so pen lifts
    never leave connecting scars.
    """
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    pen = ImageDraw.Draw(canvas)
    lengths = stroke_lengths(strokes)
    target_length = sum(lengths) * min(max(progress, 0.0), 1.0)
    for stroke, stroke_length in zip(strokes, lengths):
        if target_length >= stroke_length:
            # This stroke is finished: draw it whole, spend its length.
            pen.line(stroke, fill=(0, 0, 0), width=PEN_WIDTH, joint="curve")
            target_length -= stroke_length
        else:
            # Pen is mid-stroke: walk waypoints until the budget runs out.
            drawn_path = [stroke[0]]
            for point_a, point_b in zip(stroke[:-1], stroke[1:]):
                segment = math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
                if target_length >= segment:
                    drawn_path.append(point_b)
                    target_length -= segment
                else:
                    fraction = target_length / max(segment, 1e-6)
                    drawn_path.append((
                        point_a[0] + (point_b[0] - point_a[0]) * fraction,
                        point_a[1] + (point_b[1] - point_a[1]) * fraction,
                    ))
                    break
            if len(drawn_path) >= 2:
                pen.line(drawn_path, fill=(0, 0, 0), width=PEN_WIDTH, joint="curve")
            break
    return canvas


def digit_frame_from_canvas(canvas_image, caption, pixel_scale,
                            bar_fraction=None, bar_color=(80, 200, 120)):
    """Runs a canvas through the real app pipeline and renders the captioned cell.

    Takes the 280px RGB canvas, feeds {'composite': ...} through
    app.preprocess_canvas_drawing (border detection, 20px fit, max-norm),
    classifies the resulting 28x28 with the trained weights, and renders the
    digit plus a caption strip with a colored loading bar underneath:
    blue while the pen is still writing (bar shows stroke progress), green at
    the verdict (bar shows model confidence).
    Returns (cell_image, predicted_digit or None, confidence 0..1).
    """
    model_input = preprocess_canvas_drawing(
        {"composite": np.array(canvas_image), "layers": [], "background": None})
    if model_input is None or float(model_input.sum()) == 0.0:
        predicted_digit, confidence = None, 0.0
    else:
        probabilities = predict_proba(model_input, NETWORK_PARAMS)
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])

    digit_view = Image.fromarray((np.array(canvas_image.convert("L"))), mode="L")
    digit_view = digit_view.resize((28 * pixel_scale, 28 * pixel_scale), Image.NEAREST)
    digit_view = digit_view.convert("RGB")

    # Caption strip: text line on top, colored loading bar below it.
    strip_height = 34
    cell = Image.new("RGB", (digit_view.width, digit_view.height + strip_height),
                     (255, 255, 255))
    cell.paste(digit_view, (0, 0))
    strip_pen = ImageDraw.Draw(cell)
    strip_pen.text((6, digit_view.height + 3), caption, fill=(0, 0, 0))
    if bar_fraction is not None:
        # Track (light gray) spans the strip width; fill shows progress.
        filled_fraction = min(max(bar_fraction, 0.0), 1.0)
        track_top = digit_view.height + 22
        track_bottom = digit_view.height + 30
        strip_pen.rectangle([6, track_top, digit_view.width - 6, track_bottom],
                            fill=(225, 225, 225))
        strip_pen.rectangle([6, track_top,
                             6 + (digit_view.width - 12) * filled_fraction, track_bottom],
                            fill=bar_color)
    return cell, predicted_digit, confidence


def main():
    """Draws digits 1-9 stroke by stroke, saving portraits + slideshow + grid."""
    parser = argparse.ArgumentParser(description="Draw the README writing-digits GIFs.")
    parser.add_argument("--model", default=os.path.join("checkpoints", "model.bin"))
    parser.add_argument("--output", default=os.path.join("assets", "demo.gif"))
    parser.add_argument("--scale", type=int, default=5, help="Pixel upscale factor.")
    parser.add_argument("--frame_ms", type=int, default=120, help="Ms per frame.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: '{args.model}' not found.", file=sys.stderr)
        raise SystemExit(1)

    global NETWORK_PARAMS
    NETWORK_PARAMS = load_model(args.model)
    print(f"[+] Loaded weights from {args.model}")

    all_strokes = digit_strokes()
    per_digit_dir = os.path.dirname(args.output) or "."
    all_frames = []
    frames_by_digit = {}

    for digit in range(1, 10):
        strokes = all_strokes[digit]
        digit_frames = []
        # Reveal the pen path over REVEAL_FRAMES, blue bar tracks the pen.
        # Then hold the verdict, green bar shows the model confidence.
        reveal_total = REVEAL_FRAMES + HOLD_FRAMES
        for frame_index in range(1, REVEAL_FRAMES + 1):
            progress = frame_index / REVEAL_FRAMES
            canvas = draw_partial(strokes, progress)
            cell, _, _ = digit_frame_from_canvas(
                canvas, f"writing {digit}...", args.scale,
                bar_fraction=(frame_index / reveal_total), bar_color=(66, 135, 245))
            digit_frames.append(cell)
        # Final canvas classifies the finished drawing for the verdict line.
        finished = draw_partial(strokes, 1.0)
        _, predicted_digit, confidence = digit_frame_from_canvas(finished, "", args.scale)
        verdict = (f"{digit} -> pred {predicted_digit} ({100.0 * confidence:.0f}%)"
                   if predicted_digit is not None else f"{digit} -> no input?")
        for _ in range(HOLD_FRAMES):
            cell, _, _ = digit_frame_from_canvas(
                finished, verdict, args.scale,
                bar_fraction=confidence, bar_color=(80, 200, 120))
            digit_frames.append(cell)
        print(f"[+] Digit {digit}: final prediction "
              f"{predicted_digit} ({100.0 * confidence:.1f}%)")
        frames_by_digit[digit] = digit_frames
        all_frames.extend(digit_frames)

        portrait_path = os.path.join(per_digit_dir, f"demo_{digit}.gif")
        digit_frames[0].save(portrait_path, save_all=True,
                             append_images=digit_frames[1:],
                             duration=args.frame_ms, loop=0)
        print(f"    Saved portrait to {portrait_path}")

    # Synced 3x3 grid: step N of every digit in ONE frame, always in lockstep.
    grid_cells = [frames_by_digit[digit] for digit in range(1, 10)]
    cell_width, cell_height = grid_cells[0][0].size
    grid_frames = []
    for step_index in range(len(grid_cells[0])):
        grid_image = Image.new("RGB", (cell_width * 3, cell_height * 3), (240, 240, 240))
        for slot, digit_frames in enumerate(grid_cells):
            grid_image.paste(digit_frames[step_index],
                             ((slot % 3) * cell_width, (slot // 3) * cell_height))
        grid_frames.append(grid_image)
    grid_path = os.path.join(per_digit_dir, "demo_grid.gif")
    grid_frames[0].save(grid_path, save_all=True, append_images=grid_frames[1:],
                         duration=args.frame_ms, loop=0)
    print(f"[+] Saved synced grid GIF ({len(grid_frames)} frames) to {grid_path}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    all_frames[0].save(args.output, save_all=True, append_images=all_frames[1:],
                        duration=args.frame_ms, loop=0)
    print(f"[+] Saved slideshow GIF ({len(all_frames)} frames) to {args.output}")


if __name__ == "__main__":
    main()
