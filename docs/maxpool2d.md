# MaxPool2D (`include/maxpool2d.hpp`)

Downsamples each channel by keeping the maximum of every `pool_size × pool_size`
window stepped by `stride`. With pool 2 / stride 2, 28×28 maps become 14×14 —
small translation invariance plus ~4× less compute downstream.

## Shapes

Input `(batch, C, in_h, in_w)` → Output `(batch, C, out_h, out_w)`:

```
out_h = (in_h - pool_size) / stride + 1   (same for out_w)
```

## Forward

```
pooled[batch_index, channel_index, output_row, output_col]
    = max over window cells input_row = output_row*stride+window_row, input_col = output_col*stride+window_col
```

Each winner's flat input index is recorded in `max_indices` for backward.
`(batch, channel)` tiles are disjoint → OpenMP `collapse(2)`.

## Backward

```
input_grad[winning_index] += output_grad[output_index]   (all else stays 0)
```

With non-overlapping 2×2 windows each input pixel can win at most one window,
so accumulation (`+=`) is exact even with ties (first max wins).

## Worked example (4×4 → 2×2, pool 2 stride 2)

```
input =  1  5  2  0       windows: |1 5| |2 0|  → max 5, max 2
         3  4  6  1                  |3 4| |6 1|
         7  2  9  8                  |7 2| |9 8|  → max 7, max 9
         0  1  4  3                  |0 1| |4 3|

pooled =  5  2
          7  9
```

If upstream gradient is `[[1, 2], [3, 4]]`, the 4×4 input gradient is zero
except 1 at the 5-position, 2 at the 2-position, 3 at 7, 4 at 9.
