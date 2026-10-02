#ifndef SGD_HPP
#define SGD_HPP

// SGD: vanilla stochastic gradient descent,
// learned_value = learned_value - learning_rate * gradient_value.
// Layers expose their storage through Parameter {data, grad} pointer pairs so
// one optimizer can sweep every registered tensor with a single step() call.
// Typical loop per batch: optimizer.zero_grad() -> forward -> backward (fills
// each grad buffer) -> optimizer.step() (applies the update below).
// Math: for every registered parameter:
//   parameter_data[element_index] -= learning_rate * parameter_grad[element_index].
// See docs/sgd.md for the update rule and a one-step numeric example.

#include <vector>
#include <algorithm>

namespace neuracpp {

// Handle pairing a parameter buffer with its gradient buffer.
// Both are non-owning pointers into layer-owned storage (e.g. Linear::weights
// and Linear::grad_weights), registered once and updated in place.
struct Parameter {
    // Pointer to the learnable values (weights or biases) to update.
    std::vector<float>* data;
    // Pointer to the matching dLoss/dParam values filled by backward().
    std::vector<float>* grad;

    // Binds the pair; both pointers must stay valid while the optimizer runs.
    Parameter(std::vector<float>* parameter_data, std::vector<float>* parameter_grad)
        : data(parameter_data), grad(parameter_grad) {}
};

class SGD {
public:
    // Step size multiplying every gradient (0.01 is the trainer's default).
    float learning_rate;
    // Every registered {data, grad} pair updated by step().
    std::vector<Parameter> registered_params;

    // Builds the optimizer with the given step size.
    explicit SGD(float learning_rate = 0.01f) : learning_rate(learning_rate) {}

    // Registers one weight or bias vector plus its gradient buffer.
    // Takes mutable references to layer-owned storage and stores pointers.
    // Example: optimizer.register_parameter(fc.weights.data, fc.grad_weights.data).
    void register_parameter(std::vector<float>& parameter_data, std::vector<float>& parameter_grad) {
        registered_params.emplace_back(&parameter_data, &parameter_grad);
    }

    // Applies one gradient-descent step to every registered parameter.
    // Implements p[i] = p[i] - learning_rate * grad[i], element by element.
    void step() {
        for (auto& parameter : registered_params) {
            size_t element_count = parameter.data->size();
            for (size_t element_index = 0; element_index < element_count; ++element_index) {
                (*parameter.data)[element_index] -= learning_rate * (*parameter.grad)[element_index];
            }
        }
    }

    // Clears every registered gradient buffer to 0 before the next batch.
    // Must run before forward/backward so gradients never leak across batches.
    void zero_grad() {
        for (auto& parameter : registered_params) {
            std::fill(parameter.grad->begin(), parameter.grad->end(), 0.0f);
        }
    }
};

} // namespace neuracpp

#endif // SGD_HPP
