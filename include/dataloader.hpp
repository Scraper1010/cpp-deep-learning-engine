#ifndef DATALOADER_HPP
#define DATALOADER_HPP

// DataLoader: shuffling mini-batch sampler over a loaded MNIST split.
// It keeps a permutation of sample indices; get_batch(i) gathers the i-th
// slice of that permutation into a contiguous (batch, channels, height, width)
// tensor plus matching labels. The final batch may be smaller than the rest.
// Call shuffle_data() each epoch so the network sees a fresh random order.
// See docs/dataloader.md for the batching math and a 10-sample worked example.

#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <cstddef>
#include "tensor.hpp"
#include "mnist_reader.hpp"

namespace neuracpp {

// One training step worth of data: contiguous images plus integer labels.
struct Batch {
    // Images with shape (actual_batch_size, channels, height, width).
    Tensor images;
    // Digit target per image (0-9), same length as the batch dimension.
    std::vector<int64_t> labels;
};

class DataLoader {
public:
    // Binds the loader to loaded data without copying the pixels.
    // Takes the dataset to sample from, the rows per batch, and whether to
    // shuffle immediately (call shuffle_data() each later epoch regardless).
    DataLoader(const MNISTData& dataset, size_t batch_size, bool shuffle_on_start = true)
        : dataset_(dataset), batch_size_(batch_size), shuffle_on_start_(shuffle_on_start) {

        // Cache the dataset geometry so batch slicing is pure arithmetic.
        sample_count_ = dataset_.images.shape[0];
        channel_count_ = dataset_.images.shape.size() > 1 ? dataset_.images.shape[1] : 1;
        image_height_ = dataset_.images.shape.size() > 2 ? dataset_.images.shape[2] : 28;
        image_width_ = dataset_.images.shape.size() > 3 ? dataset_.images.shape[3] : 28;
        // Flat floats per image: channels * height * width (784 for MNIST).
        floats_per_sample_ = channel_count_ * image_height_ * image_width_;

        // Start with the identity permutation 0, 1, 2, ..., N-1.
        sample_order_.resize(sample_count_);
        std::iota(sample_order_.begin(), sample_order_.end(), 0);

        if (shuffle_on_start_) {
            shuffle_data();
        }
    }

    // Randomly reorders the sample permutation for a new epoch.
    // Uses a thread-local Mersenne Twister so consecutive epochs differ.
    void shuffle_data() {
        static std::random_device seed_source;
        static std::mt19937 random_generator(seed_source());
        std::shuffle(sample_order_.begin(), sample_order_.end(), random_generator);
    }

    // Returns the batch count, rounding up so the tail is never dropped.
    // Example: 60000 samples at batch 64 -> ceil(60000 / 64) = 938 batches.
    size_t num_batches() const {
        return (sample_count_ + batch_size_ - 1) / batch_size_;
    }

    // Returns the total sample count in the bound dataset.
    size_t num_samples() const { return sample_count_; }

    // Gathers one mini-batch by position in the current permutation.
    // Takes the zero-based batch position (0 .. num_batches() - 1).
    // Returns contiguous batch images plus the matching labels in order.
    Batch get_batch(size_t batch_position) const {
        // Slice [start, end) of the permutation for this batch.
        size_t slice_start = batch_position * batch_size_;
        size_t slice_end = std::min(slice_start + batch_size_, sample_count_);
        size_t actual_batch_size = slice_end - slice_start;

        Batch batch;
        batch.images = Tensor({actual_batch_size, channel_count_, image_height_, image_width_});
        batch.labels.resize(actual_batch_size);

        // Copy each sample's contiguous pixel block into the batch buffer.
        for (size_t slot_index = 0; slot_index < actual_batch_size; ++slot_index) {
            // Dataset row selected by the shuffled permutation.
            size_t dataset_index = sample_order_[slice_start + slot_index];

            // Source/destination pointers into the flat float buffers.
            const float* source_pixels = dataset_.images.data.data() + (dataset_index * floats_per_sample_);
            float* dest_pixels = batch.images.data.data() + (slot_index * floats_per_sample_);
            std::copy(source_pixels, source_pixels + floats_per_sample_, dest_pixels);

            // Carry the matching digit label alongside the pixels.
            batch.labels[slot_index] = dataset_.labels[dataset_index];
        }

        return batch;
    }

private:
    // The full loaded split (referenced, never copied) the batches slice from.
    const MNISTData& dataset_;
    // Requested rows per batch (the last batch may hold fewer rows).
    size_t batch_size_;
    // Whether the constructor shuffled the permutation immediately.
    bool shuffle_on_start_;
    // Total rows available in the dataset.
    size_t sample_count_;
    // Image geometry cached from the dataset tensor for fast slicing.
    size_t channel_count_, image_height_, image_width_;
    // Flat floats per image (channels * height * width).
    size_t floats_per_sample_;
    // Current permutation of dataset row indices; reshuffled each epoch.
    std::vector<size_t> sample_order_;
};

} // namespace neuracpp

#endif // DATALOADER_HPP
