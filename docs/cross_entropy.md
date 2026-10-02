# CrossEntropyLoss (`include/cross_entropy.hpp`)

Softmax plus mean negative-log-likelihood — the standard classification loss.

## Forward (per sample, then averaged)

1. **Stabilize:** `row_max = max(logit_row)`; work with `logit_row - row_max` so every `exp ≤ 1`.
2. **Exponentiate:** `shifted_exp[class_index] = exp(logit[class_index] - row_max)`, normalizer `exp_sum = sum of shifted_exp`.
3. **Normalize:** `probability[class_index] = shifted_exp[class_index] / exp_sum` (rows sum to 1).
4. **Score:** `sample_loss = -log(clamp(probability[target_digit], 1e-7, 1))`.

Batch loss = mean of the per-sample losses (batch-size independent scale).
Probabilities are cached in `probs`, targets in `targets_cache`, so
`backward()` needs no arguments.

## Backward (the short form)

```
logit_gradient[sample_index, class_index]
    = (predicted_probability - target_one_hot) / batch_size
```

i.e. predicted probability minus 1 at the true class, scaled by `1 / batch_size`.

## Worked example (3 classes, target = class 2)

```
logits = [2.0, 1.0, 0.1],  row_max = 2.0
shifted exp = [e^0, e^-1, e^-1.9] = [1.0, 0.368, 0.150],  exp_sum = 1.518,
normalize = [1.0/1.518, 0.368/1.518, 0.150/1.518] -> probabilities
probabilities = [0.659, 0.242, 0.099]
```

Why the log touches only the last value: full cross-entropy sums over every
class, `loss = -(0·log(0.659) + 0·log(0.242) + 1·log(0.099))`, but the true
distribution is one-hot — 1 at the target digit, 0 elsewhere — so every
non-target term vanishes and only `-log(0.099) = 2.31` survives. The rivals
still mattered: they fed the shared normalizer `exp_sum = 1.518` that shrank
the target's probability from `0.150` to `0.099`. Their influence flows through
the denominator, not through the log.

```
loss = -log(0.099) = 2.31
```

Where backward starts: at this scalar loss, walking home in two steps.
First through the log (only the target has a path):
`dLoss/dProbability[target] = -1 / 0.099`. Then through softmax, where that
signal fans out — pushing the target's own logit up and the rivals' logits
down, since raising a rival would steal denominator share from the target.
The two steps collapse into the short form, which is both the end of the loss
and the start of the network's backward pass (`DigitNetwork::backward()`
receives exactly this vector):

```
gradient for batch_size 1 = [0.659 - 0, 0.242 - 0, 0.099 - 1] = [0.659, 0.242, -0.901]
```

Without the max-subtraction, `exp(large logit)` overflows to infinity; with it
the math is identical (it cancels in numerator/denominator) but safe.
