# Flatten (`include/flatten.hpp`)

Reshapes `(batch, dim1, dim2, …)` into `(batch, features)` with
`features = dim1 × dim2 × …`, preserving row-major order. In the network,
`(batch, 64, 7, 7)` becomes `(batch, 3136)` so `Linear` can classify it.

## Why a plain copy works

Row-major storage already nests as channel → row → column, so collapsing the
trailing dimensions is just a reinterpretation — the code copies the buffer
verbatim (`flat_output.data = input_tensor.data`) and remembers the original
shape in `input_shape_cache`.

## Backward

Exact inverse: rebuilds the cached shape and passes every gradient value
straight through 1:1 (`input_gradient.data = output_gradient.data`).

## Worked example

```
input shape (1, 2, 2, 2), data = [a b c d e f g h]
  channel 0 rows: [a b] [c d], channel 1 rows: [e f] [g h]
forward  → shape (1, 8), data = [a b c d e f g h]
backward → shape (1, 2, 2, 2), same 8 gradient values in the same slots
```
