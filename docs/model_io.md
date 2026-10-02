# Checkpoint format (`include/model_io.hpp`)

`save_model` / `load_model` persist the four learnable layers (two
convolutions + two linears = eight tensors) to one binary file.

## Layout (little-endian)

```
bytes 0-7 : tag "NEURACPP" (8 ASCII bytes, sanity check on load)
next 14 x u64 : network shape, in order
    conv1 in/out/kernel/stride/pad   (5 values)
    conv2 in/out/kernel/stride/pad   (5 values)
    fc1 in/out, fc2 in/out           (4 values)
then 8 tensors, each (u64 count + count float32 values):
    conv1 weights, conv1 bias, conv2 weights, conv2 bias,
    fc1 weights, fc1 bias, fc2 weights, fc2 bias
```

`save_model` creates missing parent folders; `load_model` rejects bad tags,
shape mismatches (file trained by a different layout → retrain), and
truncated blobs with explicit messages.

## Worked byte example

Saving the real network writes `8 + 14*8 + 8*8` header bytes, then:

```
conv1 weights: u64 288  + 288 float32  (32 filters × 1 ch × 3 × 3)
conv1 bias:    u64 32   + 32 float32
conv2 weights: u64 18432 + 18432 float32 (64 × 32 × 3 × 3)
conv2 bias:    u64 64   + 64 float32
fc1 weights:   u64 401408 + ... (3136 × 128)
...
```

`python/neuracpp_predict.py::load_model` reads the identical layout with
`struct` + `numpy.fromfile` — see [inference](inference.md).
