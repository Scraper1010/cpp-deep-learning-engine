# DataLoader (`include/dataloader.hpp`)

Shuffling mini-batch sampler. Holds a permutation `sample_order_` of dataset
rows; `get_batch(i)` gathers the i-th slice into a contiguous tensor.

## Batching math

```
num_batches = ceil(sample_count / batch_size)
slice i covers permutation rows [i*B, min(i*B + B, N))
```

Each image occupies `floats_per_sample = channels × height × width` contiguous
floats (784 for MNIST), so gathering is a `std::copy` of that block plus its
label. The final batch may be smaller (e.g. 60000 = 937×64 + 32).

## Shuffling

`shuffle_data()` applies `std::shuffle` with a Mersenne Twister to the index
permutation — the pixel storage is never moved. Call it every epoch so batches
differ; the test loader is constructed with shuffling off for stable eval.

## Worked example (10 samples, batch 4)

```
order after shuffle: [7, 2, 9, 0, 5, 1, 8, 3, 6, 4]
batch 0 → rows 7, 2, 9, 0 (full)
batch 1 → rows 5, 1, 8, 3 (full)
batch 2 → rows 6, 4        (partial tail of 2)
```
