# Linear (`include/linear.hpp`)

Fully-connected map: every output is a weighted sum of **all** inputs plus a bias.

## Shapes

- Weights: `(in_features, out_features)` — row = input dim, column = output dim
- Bias: `(1, out_features)`
- Input: `(batch, in_features)` → Output: `(batch, out_features)`

## Forward

```
output[sample_index, output_feature]
    = bias[output_feature]
      + sum over input_feature of input[sample_index, input_feature] * weights[input_feature, output_feature]
(i.e. output_matrix = input_matrix * weight_matrix + bias)
```

Rows are independent, so the batch loop parallelizes with OpenMP.

## Backward (the three textbook products)

Given the upstream gradient (dLoss/dOutput) of shape `(batch, out_features)`:

```
grad_bias[output_feature] = sum over samples of upstream_gradient
grad_weights[input_feature, output_feature]
    = sum over samples of cached_input * upstream_gradient   (input-transpose times gradient)
input_gradient[sample_index, input_feature]
    = sum over outputs of upstream_gradient * weights        (gradient times weights-transpose)
```

Same threading strategy as Conv2D: thread-local weight/bias sums merged once,
input-gradient rows written lock-free.

## Initialization

He normal $W \sim \mathcal{N}(0, \sqrt{2 / in\_features})$, biases zero.

## Worked example (2 features → 3 outputs, batch of 1)

```
X = [2.0, -1.0]   W = [ 0.5  1.0 -0.5 ]   b = [0.1, 0.0, 0.2]
                      [ 1.0  0.0  2.0 ]

out[0] = 0.1 + 2.0*0.5 + (-1.0)*1.0 = 0.1
out[1] = 0.0 + 2.0*1.0 + (-1.0)*0.0 = 2.0
out[2] = 0.2 + 2.0*-0.5 + (-1.0)*2.0 = -2.8
```

If upstream `G = [1.0, 0.5, -1.0]`, then `dBias = G`,
`dWeight[0,:] = X[0]*G = [2.0, 1.0, -2.0]`, and so on down the rows.
