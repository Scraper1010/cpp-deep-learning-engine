#ifndef LINEAR_HPP
#define LINEAR_HPP

// Linear: fully-connected map output = input * weights + bias.
// With input rows as feature vectors:
// output[sample_index, output_feature] = bias[output_feature]
//   + sum over input_feature of
//     input[sample_index, input_feature] * weights[input_feature, output_feature].
// Weights use He initialization Normal(0, sqrt(2 / in_features)); biases are zero.
// Backward implements the three textbook products:
//   dBias    = sum over batch of grad_out,
//   dWeights = input^T * grad_out,
//   dInput   = grad_out * weights^T.
// See docs/linear.md for the matrix equations and a tiny 2x3 numeric example.

#include "layer.hpp"
#include <random>
#include <cmath>
#include <vector>

namespace neuracpp {
    class Linear : public Layer {
        public:
            // Length of each incoming feature vector (columns of the input).
            size_t in_features;
            // Length of each outgoing vector (logits or hidden activations).
            size_t out_features;

            // Weight matrix, shape (in_features, out_features), row = input dim.
            Tensor weights;
            // Per-output offset added to every sample, shape (1, out_features).
            Tensor bias;
            // Accumulated dLoss/dWeight, same shape as weights.
            Tensor grad_weights;
            // Accumulated dLoss/dBias, same shape as bias.
            Tensor grad_bias;

            // Builds the layer and He-initializes the weight matrix.
            // Takes the input vector length and the desired output length.
            Linear(size_t in_features, size_t out_features)
                :in_features(in_features), out_features(out_features),
                weights({in_features, out_features}),
                bias({1, out_features}),
                grad_weights({in_features, out_features}),
                grad_bias({1, out_features}) {

                // He/Kaiming init: variance 2/fan_in keeps ReLU nets stable.
                // Here fan_in is simply the input feature count.
                std::random_device random_seed_source;
                std::mt19937 random_generator(random_seed_source());
                float init_stddev = std::sqrt(2.0f / static_cast<float>(in_features));
                std::normal_distribution<float> init_distribution(0.0f, init_stddev);

                // Draw every weight independently from the He distribution.
                for (size_t weight_index = 0; weight_index < weights.size(); ++weight_index){
                    weights[weight_index] = init_distribution(random_generator);
                }

                // Biases start at zero; weights already break symmetry.
                bias.data.assign(out_features, 0.0f);
                }

            // Multiplies each input row by the weight matrix and adds the bias.
            // Takes input_batch with shape (batch_size, in_features).
            // Caches the input for the backward pass.
            // Returns logits with shape (batch_size, out_features).
            Tensor forward(const Tensor& input_batch) override {
                // Keep the input: backward() needs it for dLoss/dWeight.
                cached_input_ = input_batch;
                // Row count drives the loop; feature width is fixed.
                size_t batch_size = input_batch.shape[0];

                // Allocate the output rows (zero-filled, then overwritten).
                Tensor activated_output({batch_size, out_features}, 0.0f);

                // Matrix multiply: output = input * weights + bias; rows are independent -> parallel.
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static)
                #endif
                for (size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
                    for (size_t output_feature = 0; output_feature < out_features; ++output_feature){
                        // Start from this output's bias term.
                        float row_sum = bias[output_feature];
                        // Dot the input row with this output's weight column.
                        for (size_t input_feature = 0; input_feature < in_features; ++input_feature){
                            row_sum += cached_input_[sample_index * in_features + input_feature] *
                                       weights[input_feature * out_features + output_feature];
                        }
                        activated_output[sample_index * out_features + output_feature] = row_sum;
                    }
                }
                return activated_output;
            }

            // Backpropagates through the matrix multiply.
            // Takes output_gradient (dLoss/dOutput, shape (batch, out_features)).
            // Accumulates dLoss/dWeight (= X^T * grad) and dLoss/dBias (= row sums).
            // Returns dLoss/dInput (= grad * W^T) with the cached input's shape.
            Tensor backward(const Tensor& output_gradient) override {
                // Row count of the cached forward input.
                size_t batch_size = cached_input_.shape[0];

                // Fresh gradient buffers for this batch.
                grad_weights.data.assign(grad_weights.size(), 0.0f);
                grad_bias.data.assign(grad_bias.size(), 0.0f);

                // Gradient flowing to the previous layer.
                Tensor input_gradient({batch_size, in_features}, 0.0f);

                // Weight/bias grads are shared across samples -> thread-local
                // sums merged once. Input-grad rows are disjoint -> lock-free.
                #ifdef _OPENMP
                #pragma omp parallel
                {
                    // Private accumulators sized like the shared buffers.
                    std::vector<float> local_weight_grad(grad_weights.size(), 0.0f);
                    std::vector<float> local_bias_grad(grad_bias.size(), 0.0f);
                    #pragma omp for schedule(static) nowait
                    for (size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
                        for (size_t output_feature = 0; output_feature < out_features; ++output_feature) {
                            // Upstream gradient for this (sample, output) cell.
                            float upstream_grad = output_gradient[sample_index * out_features + output_feature];
                            // dLoss/dBias accumulates the upstream value.
                            local_bias_grad[output_feature] += upstream_grad;
                            for (size_t input_feature = 0; input_feature < in_features; ++input_feature) {
                                // dLoss/dWeight += input * upstream (outer product).
                                local_weight_grad[input_feature * out_features + output_feature] +=
                                    cached_input_[sample_index * in_features + input_feature] * upstream_grad;
                                // dLoss/dInput += upstream * weight (times W^T).
                                input_gradient[sample_index * in_features + input_feature] +=
                                    upstream_grad * weights[input_feature * out_features + output_feature];
                            }
                        }
                    }
                    // Merge every thread's private sums into the shared buffers.
                    #pragma omp critical
                    {
                        for (size_t grad_index = 0; grad_index < grad_weights.size(); ++grad_index) {
                            grad_weights[grad_index] += local_weight_grad[grad_index];
                        }
                        for (size_t bias_index = 0; bias_index < grad_bias.size(); ++bias_index) {
                            grad_bias[bias_index] += local_bias_grad[bias_index];
                        }
                    }
                }
                #else
                // Single-threaded reference path: identical math, direct sums.
                for (size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
                    for (size_t output_feature = 0; output_feature < out_features; ++output_feature) {
                        float upstream_grad = output_gradient[sample_index * out_features + output_feature];

                        // Gradient w.r.t bias.
                        grad_bias[output_feature] += upstream_grad;

                        for (size_t input_feature = 0; input_feature < in_features; ++input_feature) {
                            // Gradient w.r.t weights: X^T * GradOutput.
                            grad_weights[input_feature * out_features + output_feature] +=
                                cached_input_[sample_index * in_features + input_feature] * upstream_grad;

                            // Gradient w.r.t input: GradOutput * W^T.
                            input_gradient[sample_index * in_features + input_feature] +=
                                upstream_grad * weights[input_feature * out_features + output_feature];
                        }
                    }
                }
                #endif
                return input_gradient;
            }

        private:
            // Copy of the forward input batch, kept for forming gradients.
            Tensor cached_input_;
    };
}

#endif
