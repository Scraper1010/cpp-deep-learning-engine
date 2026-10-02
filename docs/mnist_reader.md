# MNISTReader (`include/mnist_reader.hpp`)

Parses the original MNIST IDX binaries into engine tensors.

## Byte layouts (all header integers big-endian)

**Images (IDX3)** — 16-byte header, then raw pixels:

```
offset 0: u32 magic = 2051
offset 4: u32 image_count   (60000 train / 10000 test)
offset 8: u32 rows (= 28)
offset 12: u32 cols (= 28)
offset 16: image_count * rows * cols bytes, pixels 0-255 row-major
```

**Labels (IDX1)** — 8-byte header, then raw digits:

```
offset 0: u32 magic = 2049
offset 4: u32 label_count  (must equal image_count)
offset 8: label_count bytes, one digit 0-9 each
```

`swap_endian()` reverses each 32-bit header word into host order. Pixels are
stored as `byte / 255.0f` in a `(N, 1, 28, 28)` tensor; labels widen to int64.

## Worked header example

```
first 16 image bytes: 00 00 08 03 | 00 00 EA 60 | 00 00 00 1C | 00 00 00 1C
magic: 0x00000803 = 2051 ✓, count: 0xEA60 = 60000, rows = cols = 0x1C = 28
```

The loader throws descriptive errors for missing files, wrong magic numbers,
or image/label count mismatches.
