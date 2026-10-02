# Layer interface (`include/layer.hpp`)

The contract every stage honors: `forward` computes, `backward` differentiates.

```cpp
class Layer {
public:
    virtual Tensor forward(const Tensor& input_tensor) = 0;
    virtual Tensor backward(const Tensor& output_gradient) = 0;
};
```

## The pattern

1. **Forward** takes an input tensor, runs the layer's math, and **caches**
   whatever `backward` will need (e.g. `Conv2D` keeps `cached_input_`,
   `MaxPool2D` keeps the winning indices, `ReLU` keeps the raw input).
2. **Backward** receives `dLoss/dOutput` from the next layer and returns
   `dLoss/dInput` for the previous layer, accumulating `grad_weights` /
   `grad_bias` as a side effect where the layer has parameters.

## Chain rule in one line

If layer output is $y = f(x)$, the layer maps the incoming gradient $g = dL/dy$
to the outgoing gradient $dL/dx = g \cdot f'(x)$ — each class below implements
its own $f'$ (see its doc page).

`Conv2D` and `Linear` inherit from `Layer`; activation/pooling/flatten stages
follow the same forward/backward convention structurally.
