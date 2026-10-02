#ifndef RELU_HPP
#define RELU_HPP

// ReLU: element-wise rectified linear unit, output = max(0, input).
// It adds the network's non-linearity: without it, stacked layers would
// collapse into a single linear map. Its derivative is 1 for positive inputs
// and 0 otherwise, so dead (negative) neurons pass no gradient.
// Math: forward(input_tensor) = max(0, input_tensor);
// backward(output_gradient) = output_gradient where input was positive, else 0.
// See docs/relu.md for the formula and a worked numeric example.

#include "tensor.hpp"
#include <algorithm>

namespace neuracpp {

class ReLU {
public:
    // Copy of the forward input, kept so backward() knows which positions
    // were positive (gradient flows) and which were clamped (gradient blocked).
    Tensor input_cache;

    // Default-constructs a stateless ReLU (cache starts empty).
    ReLU() = default;

    // Applies max(0, value) to every element of input_tensor.
    // Caches a copy of input_tensor for the backward pass.
    // Returns a new tensor with the same shape, all negatives replaced by 0.
    Tensor forward(const Tensor& input_tensor) {
        // Remember the raw input so backward() can test each position for > 0.
        input_cache = input_tensor;
        // Allocate the output with the same logical shape (defaults to zeros).
        Tensor activated_output(input_tensor.shape);

        // Total flat element count to process.
        size_t total_elements = input_tensor.data.size();
        // Each element is independent -> safe to split across CPU threads.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t element_index = 0; element_index < total_elements; ++element_index) {
            // Clamp negatives to zero, keep positives unchanged.
            activated_output.data[element_index] = std::max(0.0f, input_tensor.data[element_index]);
        }

        return activated_output;
    }

    // Routes the incoming gradient only through positions that were positive.
    // Takes output_gradient (dLoss/dOutput, same shape as forward's output).
    // Returns dLoss/dInput: output_gradient where cached input was positive,
    // otherwise 0 (gradient is blocked by the flat part of ReLU).
    Tensor backward(const Tensor& output_gradient) {
        // Gradient tensor has the same shape as the layer's input.
        Tensor input_gradient(output_gradient.shape);

        // Element count must match the cached forward input.
        size_t total_elements = output_gradient.data.size();
        // Element-wise mask -> embarrassingly parallel.
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t element_index = 0; element_index < total_elements; ++element_index) {
            // Derivative of max(0,x) is 1 for x > 0, else 0.
            input_gradient.data[element_index] =
                (input_cache.data[element_index] > 0.0f) ? output_gradient.data[element_index] : 0.0f;
        }

        return input_gradient;
    }
};

} // namespace neuracpp

#endif // RELU_HPP
