#ifndef MAXPOOL2D_HPP
#define MAXPOOL2D_HPP

// MaxPool2D: downsamples each channel by keeping the maximum of every
// pool_size x pool_size window, stepped by stride.
// For MNIST: (B, C, 28, 28) with pool 2 / stride 2 becomes (B, C, 14, 14).
// This gives small translation invariance and cuts later compute by ~4x.
// Backward routes each output gradient to the single winning input pixel
// recorded during forward (all other pixels in the window get 0).
// See docs/maxpool2d.md for the formula and a worked 4x4 -> 2x2 example.

#include "tensor.hpp"
#include <vector>
#include <limits>
#include <algorithm>

namespace neuracpp {

class MaxPool2D {
public:
    // Height and width of the square pooling window (2 means 2x2 windows).
    size_t pool_size;
    // Step between window positions (2 means non-overlapping windows).
    size_t stride;

    // Copy of the forward input, kept so backward() can size its output.
    Tensor input_cache;
    // For every flat output index, the flat input index that held the maximum.
    // Recorded in forward(), consumed in backward() to route gradients.
    std::vector<size_t> max_indices;

    // Builds a pooling layer; defaults match the classic 2x2 stride-2 halving.
    explicit MaxPool2D(size_t pool_size = 2, size_t stride = 2)
        : pool_size(pool_size), stride(stride) {}

    // Keeps the max of each window; records each winner's input index.
    // Takes input_tensor with shape (batch, channels, height, width).
    // Returns the downsampled tensor with shape
    // (batch, channels, height_out, width_out) where
    // height_out = (height - pool_size) / stride + 1 (same for width).
    Tensor forward(const Tensor& input_tensor) {
        // Cache the input for shape reference in backward().
        input_cache = input_tensor;

        // Unpack the four input dimensions with descriptive names.
        size_t batch_size = input_tensor.shape[0];
        size_t channel_count = input_tensor.shape[1];
        size_t input_height = input_tensor.shape[2];
        size_t input_width = input_tensor.shape[3];

        // Output spatial size shrinks according to window and stride.
        size_t output_height = (input_height - pool_size) / stride + 1;
        size_t output_width = (input_width - pool_size) / stride + 1;

        // Allocate the downsampled output and the winner-index table.
        Tensor pooled_output({batch_size, channel_count, output_height, output_width});
        max_indices.resize(pooled_output.data.size());

        // Each (batch, channel) tile is independent -> parallelize over both.
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
            for (size_t channel_index = 0; channel_index < channel_count; ++channel_index) {
                // Slide the window over every output position.
                for (size_t output_row = 0; output_row < output_height; ++output_row) {
                    for (size_t output_col = 0; output_col < output_width; ++output_col) {
                        // Flat index of this output element (NCHW layout).
                        size_t output_index =
                            ((batch_index * channel_count + channel_index) * output_height + output_row) * output_width + output_col;
                        // Track the largest value and where it came from.
                        float best_value = -std::numeric_limits<float>::infinity();
                        size_t best_input_index = 0;

                        // Scan the pool_size x pool_size input window.
                        for (size_t window_row = 0; window_row < pool_size; ++window_row) {
                            for (size_t window_col = 0; window_col < pool_size; ++window_col) {
                                // Input pixel covered by this window cell.
                                size_t input_row = output_row * stride + window_row;
                                size_t input_col = output_col * stride + window_col;
                                size_t input_index =
                                    ((batch_index * channel_count + channel_index) * input_height + input_row) * input_width + input_col;

                                // Keep the larger value and remember its position.
                                if (input_tensor.data[input_index] > best_value) {
                                    best_value = input_tensor.data[input_index];
                                    best_input_index = input_index;
                                }
                            }
                        }

                        // Store the winner and where to send its gradient later.
                        pooled_output.data[output_index] = best_value;
                        max_indices[output_index] = best_input_index;
                    }
                }
            }
        }

        return pooled_output;
    }

    // Sends each output gradient to its recorded winning input pixel.
    // Takes output_gradient (dLoss/dPooled, same shape as forward's output).
    // Returns dLoss/dInput (zeros everywhere except winner positions, which
    // accumulate their window's gradient; ties recorded first-win).
    Tensor backward(const Tensor& output_gradient) {
        // Gradient has the input's shape, starting from all zeros.
        Tensor input_gradient(input_cache.shape);
        std::fill(input_gradient.data.begin(), input_gradient.data.end(), 0.0f);

        // Number of pooled (output) elements to route back.
        size_t output_element_count = output_gradient.data.size();
        for (size_t output_index = 0; output_index < output_element_count; ++output_index) {
            // Look up which input pixel won this window during forward().
            size_t winning_input_index = max_indices[output_index];
            // Accumulate (+=, safe for overlapping windows) the gradient there.
            input_gradient.data[winning_input_index] += output_gradient.data[output_index];
        }

        return input_gradient;
    }
};

} // namespace neuracpp

#endif // MAXPOOL2D_HPP
