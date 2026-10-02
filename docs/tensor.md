# Tensor (`include/tensor.hpp`)

The single data container of the engine: a flat row-major `float` buffer plus a
shape vector that gives it meaning. Everything — images, feature maps, weight
matrices, gradients — is a `Tensor`.

## Layout

- `data`: all elements in one contiguous `vector<float>`, **row-major** (the last
  dimension varies fastest).
- `shape`: logical dimensions, e.g. `{64, 1, 28, 28}` for a batch of MNIST images.

The flat index of element `(b, c, h, w)` in an NCHW tensor is:

```
flat = ((b * C + c) * H + h) * W + w
```

## Key operations

| Code | Math |
|---|---|
| `Tensor(shape, fill)` | Allocates `size_from_shape(shape)` floats set to `fill` |
| `size_from_shape(shape)` | $\prod dim_i$, e.g. `{64,1,28,28} → 50176` |
| `tensor[i]` | Direct flat access (mutable and const overloads) |
| `tensor.size()` | Total element count (`data.size()`) |

## Worked example

```cpp
Tensor images({2, 1, 2, 2}, 0.0f); // 2 grayscale 2x2 images, all zeros
// images.data holds 2*1*2*2 = 8 floats:
// [img0 r0c0, r0c1, r1c0, r1c1, img1 r0c0, r0c1, r1c0, r1c1]
images[5] = 1.0f; // second image, first row, second column
```

Row-major means the inner (width) loop walks neighboring memory — this is why
`Flatten` can reshape with a plain buffer copy.
