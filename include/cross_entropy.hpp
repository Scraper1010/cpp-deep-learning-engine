#ifndef CROSS_ENTROPY_HPP
#define CROSS_ENTROPY_HPP

// CrossEntropyLoss: softmax + mean negative-log-likelihood for classification.
// Forward turns each logit row into probabilities with a numerically stable
// softmax (subtract the row max before exp), then averages
// -log(probability_of_target_digit) over the batch. Probabilities are clamped
// to [1e-7, 1] before the log so a confident-wrong prediction costs
// -log(1e-7) instead of infinity.
// Backward is the famous short form:
// logit_gradient[sample_index, class_index]
//   = (predicted_probability - target_one_hot) / batch_size,
// where target_one_hot is 1 at the true digit and 0 elsewhere.
// Math: probability[class_index]
//         = exp(logit[class_index] - row_max)
//           / sum over every class of exp(logit - row_max);
//       loss = mean over samples of -log(probability of the sample's target);
//       logit_gradient = (predicted_probability - target_one_hot) / batch_size.
// See docs/cross_entropy.md for the derivation and a 3-class numeric example.

#include "tensor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace neuracpp {

class CrossEntropyLoss {
public:
    // Softmax probabilities from the last forward(), shape (batch, classes).
    // Kept so backward() needs no arguments.
    Tensor probs;
    // Target class per sample, copied during forward() for backward().
    std::vector<int64_t> targets_cache;

    // Default-constructs an empty loss (caches start empty).
    CrossEntropyLoss() = default;

    // Computes the mean cross-entropy over the batch.
    // Takes logit_batch with shape (batch_size, class_count) and the matching
    // target digit per row (0-9 for MNIST). Caches both for backward().
    // Returns the batch-mean loss as a single float.
    float forward(const Tensor& logit_batch, const std::vector<int64_t>& target_digits) {
        // Remember targets and unpack the batch geometry.
        targets_cache = target_digits;
        size_t batch_size = logit_batch.shape[0];
        // Class count K (10 digits for MNIST).
        size_t class_count = logit_batch.shape[1];

        // Storage for this batch's softmax rows.
        probs = Tensor(logit_batch.shape);
        // Running sum of per-sample losses, divided by batch at the end.
        float total_loss = 0.0f;

        for (size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
            // Step 1: row maximum for numerical stability (exp(x - max) <= 1).
            float row_max_logit = -std::numeric_limits<float>::infinity();
            for (size_t class_index = 0; class_index < class_count; ++class_index) {
                row_max_logit = std::max(row_max_logit, logit_batch.data[sample_index * class_count + class_index]);
            }

            // Step 2: exponentiate the shifted logits, track the normalizer.
            float exp_sum = 0.0f;
            for (size_t class_index = 0; class_index < class_count; ++class_index) {
                float shifted_exp = std::exp(logit_batch.data[sample_index * class_count + class_index] - row_max_logit);
                probs.data[sample_index * class_count + class_index] = shifted_exp;
                exp_sum += shifted_exp;
            }

            // Step 3: divide by the normalizer -> true probabilities summing to 1.
            for (size_t class_index = 0; class_index < class_count; ++class_index) {
                probs.data[sample_index * class_count + class_index] /= exp_sum;
            }

            // Step 4: negative log of the target class's probability.
            int64_t target_class = target_digits[sample_index];
            float target_probability = std::clamp(
                probs.data[sample_index * class_count + static_cast<size_t>(target_class)], 1e-7f, 1.0f);
            total_loss += -std::log(target_probability);
        }

        // Mean over the batch (keeps the loss scale batch-size independent).
        return total_loss / static_cast<float>(batch_size);
    }

    // Convenience overload accepting plain int targets (e.g. from tests).
    // Converts to int64 and delegates to the main forward() above.
    float forward(const Tensor& logit_batch, const std::vector<int>& target_digits) {
        std::vector<int64_t> converted_targets(target_digits.begin(), target_digits.end());
        return forward(logit_batch, converted_targets);
    }

    // Returns dLoss/dLogits with shape (batch_size, class_count).
    // Closed form per row: (predicted probability - target one-hot) / batch_size.
    Tensor backward() {
        // Geometry of the cached softmax table.
        size_t batch_size = probs.shape[0];
        size_t class_count = probs.shape[1];

        // Gradient tensor matching the logits' shape.
        Tensor logit_gradient(probs.shape);
        // Mean (not sum) loss -> every row's gradient is divided by batch size.
        float inverse_batch_size = 1.0f / static_cast<float>(batch_size);

        for (size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
            int64_t target_class = targets_cache[sample_index];
            for (size_t class_index = 0; class_index < class_count; ++class_index) {
                size_t flat_index = sample_index * class_count + class_index;
                float predicted_prob = probs.data[flat_index];
                // One-hot target: 1 at the true digit, 0 elsewhere.
                float one_hot_target = (static_cast<int64_t>(class_index) == target_class) ? 1.0f : 0.0f;

                logit_gradient.data[flat_index] = (predicted_prob - one_hot_target) * inverse_batch_size;
            }
        }

        return logit_gradient;
    }
};

} // namespace neuracpp

#endif // CROSS_ENTROPY_HPP
