#ifndef FLATTEN_HPP
#define FLATTEN_HPP

// Flatten: reshapes (batch, channels, height, width) into (batch, features)
// where features = channels * height * width, preserving row-major order.
// Example: (B, 64, 7, 7) becomes (B, 3136) so a Linear layer can classify it.
// Backward is the exact inverse: it restores the cached 4D shape, passing
// every gradient value straight through to its original position.
// See docs/flatten.md for the ordering math and a worked example.

#include "tensor.hpp"
#include <vector>
#include <numeric>

namespace neuracpp {

class Flatten {
public:
    // Shape seen during forward(), e.g. {batch, 64, 7, 7}, so backward()
    // knows what 4D shape to restore the flat gradients into.
    std::vector<size_t> input_shape_cache;

    // Default-constructs an empty flatten stage (cache starts empty).
    Flatten() = default;

    // Collapses all dimensions after batch into one feature dimension.
    // Takes input_tensor with shape (batch, dim1, dim2, ...) of any rank >= 2.
    // Returns a (batch, features) tensor sharing the same flat data order.
    Tensor forward(const Tensor& input_tensor) {
        // Remember the full input shape for the backward reshape.
        input_shape_cache = input_tensor.shape;

        // Batch size stays as the first dimension.
        size_t batch_size = input_tensor.shape[0];
        // Multiply every non-batch dimension into one feature count.
        size_t flattened_features = 1;
        for (size_t dimension_index = 1; dimension_index < input_tensor.shape.size(); ++dimension_index) {
            flattened_features *= input_tensor.shape[dimension_index];
        }

        // Allocate the flat view and copy the contiguous buffer verbatim:
        // row-major order already matches (channel, row, col) nesting.
        Tensor flat_output({batch_size, flattened_features});
        flat_output.data = input_tensor.data;

        return flat_output;
    }

    // Restores the cached higher-rank shape, passing gradients through 1:1.
    // Takes output_gradient (dLoss/dFlat, shape (batch, features)).
    // Returns dLoss/dInput with the cached original shape and identical data.
    Tensor backward(const Tensor& output_gradient) {
        // Rebuild the original tensor shape from the forward cache.
        Tensor input_gradient(input_shape_cache);
        // Every flat gradient belongs to exactly one original position.
        input_gradient.data = output_gradient.data;

        return input_gradient;
    }
};

} // namespace neuracpp

#endif // FLATTEN_HPP
