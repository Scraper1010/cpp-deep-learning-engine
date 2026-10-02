#ifndef LAYER_HPP
#define LAYER_HPP

// Layer: the abstract interface every learnable (or fixed) stage implements.
// Each layer caches whatever its backward pass needs during forward, then maps
// an incoming loss gradient back to a gradient over its own input.
// See docs/layer.md for the design rationale.

#include "tensor.hpp"

namespace neuracpp {

class Layer {
public:
    // Destroys the layer through a base-class pointer safely.
    virtual ~Layer() = default;

    // Runs the layer's math on input_tensor and caches what backward() needs.
    // Returns the layer's output tensor (e.g. convolved feature maps).
    virtual Tensor forward(const Tensor& input_tensor) = 0;

    // Maps the gradient arriving from the next layer (output_gradient) back to
    // a gradient over this layer's input, accumulating parameter gradients
    // (grad_weights / grad_bias) as a side effect where applicable.
    // Returns the gradient with respect to this layer's input tensor.
    virtual Tensor backward(const Tensor& output_gradient) = 0;
};

} // namespace neuracpp

#endif // LAYER_HPP
