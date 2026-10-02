#ifndef TENSOR_HPP
#define TENSOR_HPP

// Tensor: the single data container of the whole engine.
// A Tensor is a flat row-major float buffer plus a shape vector that says
// how to interpret it, e.g. shape {batch, channels, height, width} for images
// or {batch, features} for classifier vectors.
// See docs/tensor.md for the math and worked indexing examples.

#include <vector>
#include <cstddef>

namespace neuracpp {

struct Tensor {
    // Flat storage for every element, row-major (last dimension varies fastest).
    std::vector<float> data;
    // Logical dimensions, e.g. {64, 1, 28, 28} for a batch of MNIST images.
    std::vector<size_t> shape;

    // Default-constructs an empty tensor (no storage, no shape).
    Tensor() = default;

    // Allocates continuous memory for the given shape, filled with fill_value.
    // Example: Tensor({2, 3}, 0.0f) holds 6 zeros (2 rows x 3 columns).
    Tensor(const std::vector<size_t>& shape, float fill_value = 0.0f)
        : data(size_from_shape(shape), fill_value), shape(shape) {}

    // Direct flat-array read/write access: tensor[element_index].
    float& operator[](size_t element_index) { return data[element_index]; }
    // Read-only flat-array access for const tensors.
    const float& operator[](size_t element_index) const { return data[element_index]; }

    // Returns the total element count (== data.size()).
    size_t size() const { return data.size(); }

    // Multiplies all dimensions to get the element count.
    // Example: size_from_shape({64, 1, 28, 28}) == 64 * 1 * 28 * 28 == 50176.
    static size_t size_from_shape(const std::vector<size_t>& shape) {
        // Start at 1 so an empty shape would mean a single scalar value.
        size_t total_elements = 1;
        // Multiply every dimension into the running total.
        for (size_t dimension : shape) total_elements *= dimension;
        return total_elements;
    }
};

} // namespace neuracpp

#endif // TENSOR_HPP
