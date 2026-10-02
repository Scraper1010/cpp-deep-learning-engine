# Conv2D (`include/conv2d.hpp`)

Sliding-window feature extractor. Each of the `out_channels` filters combines
every input channel through a `kernel_size × kernel_size` window.

## Shapes

- Weights: `(out_channels, in_channels, kernel, kernel)`
- Bias: `(1, out_channels)` — one offset per filter
- Input: `(batch, in_channels, in_h, in_w)` → Output: `(batch, out_channels, out_h, out_w)`

```
out_h = (in_h + 2*padding - kernel) / stride + 1   (same for out_w)
```

With kernel 3, stride 1, padding 1, a 28×28 map stays 28×28.

## Forward

```
output[sample_index, filter_index, output_row, output_col] = bias[filter_index]
    + sum over input_channel, kernel_row, kernel_col of
      input[sample_index, input_channel, output_row*stride+kernel_row-pad, output_col*stride+kernel_col-pad]
      * weights[filter_index, input_channel, kernel_row, kernel_col]
```

Taps falling outside the image read implicit zeros (padding). Each
`(sample, filter)` tile writes disjoint outputs, so the loop parallelizes
with OpenMP `collapse(2)`.

## Backward

```
grad_bias[filter_index] += sum over samples and positions of upstream_gradient
grad_weights[weight_index] += sum over samples and positions of cached_input * upstream_gradient
input_gradient[input_index] += sum of weights * upstream_gradient (routed back through the window)
```

Weight/bias gradients are shared across the batch, so threads accumulate into
private buffers merged once (`critical`); input-gradient tiles are disjoint
per sample and need no lock. Correct for mean loss because the loss gradient
arrives pre-divided by batch size.

## Initialization

He/Kaiming normal: $W \sim \mathcal{N}(0, \sqrt{2 / fan_{in}})$ with
$fan_{in} = in\_channels \cdot kernel^2$; biases start at zero.

## Worked example (4×4 input, 3×3 filter, stride 1, pad 0 → 2×2 output)

```
input =  1  2  3  4        filter = 1 0 -1
         5  6  7  8                 1 0 -1
         9 10 11 12                 1 0 -1
        13 14 15 16          bias = 0

out[0,0] = (1*1 + 2*0 + 3*-1) + (5*1 + 6*0 + 7*-1) + (9*1 + 10*0 + 11*-1)
         = (1 - 3) + (5 - 7) + (9 - 11) = -6   (a vertical-edge response)
```

Slide the window right/down for the other three outputs. This is exactly what
the `window_sum` loop in `forward()` computes per position.
