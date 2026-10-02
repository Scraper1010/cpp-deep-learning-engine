# ReLU (`include/relu.hpp`)

Element-wise rectifier: `forward(x) = max(0, x)`. The network's non-linearity —
without it, stacked layers would collapse into one linear map.

## Forward / backward

```
forward:  activated[element_index] = max(0, input_tensor[element_index])   (negatives clamped to 0)
backward: input_gradient[element_index] = output_gradient[element_index] if input_tensor[element_index] > 0 else 0
```

The derivative is 1 on the positive branch and 0 on the flat branch, so dead
(negative) neurons pass no gradient. The forward input is cached (`input_cache`)
purely to evaluate that `x > 0` mask. Both passes are element-wise and OpenMP
parallelized.

## Worked example

```
x  = [-2.0, -0.1, 0.0, 0.5, 3.0]
y  = [ 0.0,  0.0, 0.0, 0.5, 3.0]
dy = [ 1.0,  1.0, 1.0, 1.0, 1.0]
dx = [ 0.0,  0.0, 0.0, 1.0, 1.0]   (only x > 0 passes gradient)
```

Note the boundary choice: `x == 0` blocks gradient (`> 0.0f` test in code).
