"""NumPy twin of the NeuraCPP forward pass (math reference).

Architecture (must match src/main.cpp and DigitNetwork):
  Conv2D(1->32, 3x3, stride 1, pad 1) -> ReLU -> MaxPool(2x2 stride 2)
  Conv2D(32->64, 3x3, stride 1, pad 1) -> ReLU -> MaxPool(2x2 stride 2)
  Flatten(64*7*7 = 3136) -> Linear(3136->128) -> ReLU -> Linear(128->10)
  -> softmax over the 10 digit logits.

Checkpoint layout (see include/model_io.hpp, little-endian):
  8 tag bytes "NEURACPP",
  14 x u64 network shape
    (conv1 in/out/kernel/stride/pad, conv2 in/out/kernel/stride/pad,
     fc1 in/out, fc2 in/out),
  then 8 tensors, each (u64 element count + float32 values):
    conv1 weights, conv1 bias, conv2 weights, conv2 bias,
    fc1 weights, fc1 bias, fc2 weights, fc2 bias.

Every function below mirrors its C++ header one-to-one, with the same math,
so readers can cross-check C++ against NumPy line by line.
"""

import struct
import numpy as np

# First 8 bytes of every checkpoint file ("NEURACPP" in ASCII).
CHECKPOINT_TAG = b"NEURACPP"


def load_model(checkpoint_path):
    """Reads a checkpoint file into a dict of NumPy weight arrays.

    Takes the checkpoint path written by the C++ trainer.
    Returns a dict with keys c1_w/c1_b/c2_w/c2_b/f1_w/f1_b/f2_w/f2_b
    (reshaped weight matrices plus bias vectors) and the conv
    stride/padding values needed by conv2d_forward.
    Raises ValueError on bad tags, shape mismatches, or truncated data.
    """
    with open(checkpoint_path, "rb") as checkpoint_file:
        # The file must open with the identification tag.
        tag_bytes = checkpoint_file.read(8)
        if tag_bytes != CHECKPOINT_TAG:
            raise ValueError(f"Bad tag {tag_bytes!r} in {checkpoint_path} (expected {CHECKPOINT_TAG!r})")

        # Fourteen u64 shape values in the documented order.
        arch_values = struct.unpack("<14Q", checkpoint_file.read(14 * 8))
        (conv1_in, conv1_out, conv1_kernel, conv1_stride, conv1_pad,
         conv2_in, conv2_out, conv2_kernel, conv2_stride, conv2_pad,
         fc1_in, fc1_out, fc2_in, fc2_out) = arch_values

        def read_tensor(tensor_name):
            """Reads one (u64 count + float32 values) blob from the stream."""
            (element_count,) = struct.unpack("<Q", checkpoint_file.read(8))
            if element_count > (1 << 30):
                raise ValueError(f"Impossible size for {tensor_name}: {element_count}")
            values = np.fromfile(checkpoint_file, dtype="<f4", count=element_count)
            if values.size != element_count:
                raise ValueError(f"Truncated {tensor_name}: got {values.size}/{element_count}")
            return values.astype(np.float32)

        # Reshape flat blobs into the same logical shapes as the C++ tensors.
        conv1_weights = read_tensor("conv1 weights").reshape(conv1_out, conv1_in, conv1_kernel, conv1_kernel)
        conv1_bias = read_tensor("conv1 bias")
        conv2_weights = read_tensor("conv2 weights").reshape(conv2_out, conv2_in, conv2_kernel, conv2_kernel)
        conv2_bias = read_tensor("conv2 bias")
        fc1_weights = read_tensor("fc1 weights").reshape(fc1_in, fc1_out)
        fc1_bias = read_tensor("fc1 bias")
        fc2_weights = read_tensor("fc2 weights").reshape(fc2_in, fc2_out)
        fc2_bias = read_tensor("fc2 bias")

    return {
        "c1_w": conv1_weights, "c1_b": conv1_bias,
        "c2_w": conv2_weights, "c2_b": conv2_bias,
        "f1_w": fc1_weights, "f1_b": fc1_bias,
        "f2_w": fc2_weights, "f2_b": fc2_bias,
        "c1_s": conv1_stride, "c1_p": conv1_pad,
        "c2_s": conv2_stride, "c2_p": conv2_pad,
    }


def conv2d_forward(input_batch, weight_filters, bias_vector, stride=1, padding=1):
    """Slides each filter over the batch (mirrors Conv2D::forward).

    Takes input_batch (batch, in_channels, height, width), weight_filters
    (out_channels, in_channels, kernel, kernel), and one bias per filter.
    Returns feature maps (batch, out_channels, out_height, out_width) where
    out = (in + 2 * padding - kernel) // stride + 1 per spatial axis.
    """
    batch_size, _, input_height, input_width = input_batch.shape
    out_channels, _, kernel_size, _ = weight_filters.shape
    # Output spatial size from the convolution arithmetic.
    output_height = (input_height + 2 * padding - kernel_size) // stride + 1
    output_width = (input_width + 2 * padding - kernel_size) // stride + 1
    # Explicit zero padding matches the C++ border-skipping branches.
    if padding:
        input_batch = np.pad(input_batch, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    output_maps = np.zeros((batch_size, out_channels, output_height, output_width), dtype=np.float32)
    for output_row in range(output_height):
        for output_col in range(output_width):
            # The input window every filter sees at this position.
            input_window = input_batch[
                :, :,
                output_row * stride:output_row * stride + kernel_size,
                output_col * stride:output_col * stride + kernel_size,
            ]
            # One tensordot per position: dot(window, filters) + bias.
            output_maps[:, :, output_row, output_col] = bias_vector + np.tensordot(
                input_window, weight_filters, axes=([1, 2, 3], [1, 2, 3])
            )
    return output_maps


def relu(activation_batch):
    """Applies max(0, x) element-wise (mirrors ReLU::forward)."""
    return np.maximum(activation_batch, 0.0).astype(np.float32)


def maxpool2d(feature_batch, window_size=2, stride=2):
    """Keeps the max of each window (mirrors MaxPool2D::forward)."""
    batch_size, channel_count, input_height, input_width = feature_batch.shape
    output_height = (input_height - window_size) // stride + 1
    output_width = (input_width - window_size) // stride + 1
    pooled = np.zeros((batch_size, channel_count, output_height, output_width), dtype=np.float32)
    for output_row in range(output_height):
        for output_col in range(output_width):
            window = feature_batch[
                :, :,
                output_row * stride:output_row * stride + window_size,
                output_col * stride:output_col * stride + window_size,
            ]
            pooled[:, :, output_row, output_col] = window.max(axis=(2, 3))
    return pooled


def softmax(logit_batch):
    """Turns logit rows into probabilities (row max subtracted for safety)."""
    shifted = logit_batch - logit_batch.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def predict_proba(grayscale_image, network_params):
    """Runs one 28x28 image through the full network.

    Takes a (28, 28) float32 image in [0, 1] (white digit on black) and the
    parameter dict from load_model.
    Returns a (10,) probability vector over the digits.
    """
    hidden = grayscale_image.astype(np.float32).reshape(1, 1, 28, 28)
    hidden = conv2d_forward(hidden, network_params["c1_w"], network_params["c1_b"],
                            stride=int(network_params["c1_s"]), padding=int(network_params["c1_p"]))
    hidden = relu(hidden)
    hidden = maxpool2d(hidden, window_size=2, stride=2)
    hidden = conv2d_forward(hidden, network_params["c2_w"], network_params["c2_b"],
                            stride=int(network_params["c2_s"]), padding=int(network_params["c2_p"]))
    hidden = relu(hidden)
    hidden = maxpool2d(hidden, window_size=2, stride=2)
    # Flatten in C order, identical to the C++ contiguous copy.
    hidden = hidden.reshape(hidden.shape[0], -1)
    hidden = relu(hidden @ network_params["f1_w"] + network_params["f1_b"])
    logits = hidden @ network_params["f2_w"] + network_params["f2_b"]
    return softmax(logits)[0]


def predict_digit(grayscale_image, network_params):
    """Returns (best_digit, {digit: probability}) for one image."""
    probabilities = predict_proba(grayscale_image, network_params)
    return int(np.argmax(probabilities)), {str(digit): float(probabilities[digit]) for digit in range(10)}
