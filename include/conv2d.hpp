#ifndef CONV2D_HPP
#define CONV2D_HPP

// Conv2D: sliding-window feature extractor.
// Each of the out_channels filters looks at every input channel through a
// kernel_size x kernel_size window, stepped by stride, with padding rows/cols
// of implicit zeros around the input so spatial size can be preserved.
// Output size: out = (in + 2*padding - kernel) / stride + 1 (per axis).
// Example: 28x28 input, kernel 3, stride 1, padding 1 -> 28x28 output.
// Weights use He initialization: Normal(0, sqrt(2 / fan_in)) so ReLU
// activations start with stable variance. Biases start at zero.
// Forward:  output[sample_index, filter_index, output_row, output_col]
//           = bias[filter_index]
//             + sum over input_channel, kernel_row, kernel_col of
//               input[sample_index, input_channel, input_row, input_col]
//               * weights[filter_index, input_channel, kernel_row, kernel_col].
// Backward: grad_weights[weight_index] += cached_input[input_index] * upstream_grad
//           (summed over samples + positions),
//           grad_bias[filter_index] += upstream_grad (summed the same way),
//           input_gradient[input_index] += weights[weight_index] * upstream_grad
//           (routed back through the window).
// See docs/conv2d.md for the full equations and a worked 4x4 numeric example.

#include "layer.hpp"
#include <random>
#include <cmath>
#include <vector>

namespace neuracpp {

class Conv2D : public Layer {
public:
    // Number of input feature maps (1 for grayscale MNIST, 32 between blocks).
    size_t in_channels;
    // Number of learned filters = number of output feature maps.
    size_t out_channels;
    // Height and width of each square filter (3 means 3x3).
    size_t kernel_size;
    // Step of the sliding window (1 = dense, 2 = skip every other position).
    size_t stride;
    // Ring of implicit zeros around the input (1 keeps 3x3 stride-1 size).
    size_t padding;

    // Learned filters, shape (out_channels, in_channels, kernel, kernel).
    Tensor weights;
    // Per-filter offset added to every position, shape (1, out_channels).
    Tensor bias;
    // Accumulated dLoss/dWeight, same shape as weights (reset each batch).
    Tensor grad_weights;
    // Accumulated dLoss/dBias, same shape as bias (reset each batch).
    Tensor grad_bias;

    // Builds the layer and He-initializes the filters.
    // Takes input channel count, output channel count (filter count),
    // square kernel size, stride, and padding (all default to dense 3x3 kept-size use).
    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0)
        : in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), padding(padding),
          weights({out_channels, in_channels, kernel_size, kernel_size}),
          bias({1, out_channels}),
          grad_weights({out_channels, in_channels, kernel_size, kernel_size}),
          grad_bias({1, out_channels}) {

        // He/Kaiming init: variance 2/fan_in keeps ReLU activations alive.
        // fan_in = inputs feeding one output = in_channels * kernel * kernel.
        std::random_device random_seed_source;
        std::mt19937 random_generator(random_seed_source());
        float fan_in = static_cast<float>(in_channels * kernel_size * kernel_size);
        float init_stddev = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> init_distribution(0.0f, init_stddev);

        // Draw every filter weight independently from the He distribution.
        for (size_t weight_index = 0; weight_index < weights.size(); ++weight_index) {
            weights[weight_index] = init_distribution(random_generator);
        }
        // Biases start at zero (symmetric filters break symmetry already).
        bias.data.assign(out_channels, 0.0f);
    }

    // Slides every filter over the batch and adds its bias.
    // Takes input_batch with shape (batch, in_channels, in_height, in_width).
    // Caches a copy of the input for the backward pass.
    // Returns feature maps with shape (batch, out_channels, out_height, out_width).
    Tensor forward(const Tensor& input_batch) override {
        // Keep the input: backward() needs raw pixels to form weight gradients.
        cached_input_ = input_batch;

        // Unpack the batch geometry with descriptive names.
        size_t batch_size = input_batch.shape[0];
        size_t input_height = input_batch.shape[2];
        size_t input_width = input_batch.shape[3];

        // Output spatial size from the convolution arithmetic.
        size_t output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
        size_t output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

        // Allocate the output feature maps (zero-filled, then overwritten).
        Tensor output_maps({batch_size, out_channels, output_height, output_width}, 0.0f);

        // Each (sample, filter) tile writes disjoint outputs -> parallelize.
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
            for (size_t filter_index = 0; filter_index < out_channels; ++filter_index) {
                // This filter's bias seeds every one of its output positions.
                float filter_bias = bias[filter_index];

                // Visit every output position of this feature map.
                for (size_t output_row = 0; output_row < output_height; ++output_row) {
                    for (size_t output_col = 0; output_col < output_width; ++output_col) {

                        // Accumulate bias + filter dot input-window.
                        float window_sum = filter_bias;

                        // Sum over input channels and kernel taps.
                        for (size_t input_channel = 0; input_channel < in_channels; ++input_channel) {
                            for (size_t kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
                                for (size_t kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {

                                    // Map output position back to input coordinates.
                                    int input_row = static_cast<int>(output_row * stride + kernel_row) - static_cast<int>(padding);
                                    int input_col = static_cast<int>(output_col * stride + kernel_col) - static_cast<int>(padding);

                                    // Zero padding: skip taps falling outside the image.
                                    if (input_row >= 0 && input_row < static_cast<int>(input_height) &&
                                        input_col >= 0 && input_col < static_cast<int>(input_width)) {

                                        // NCHW flat index into the input sample.
                                        size_t input_index =
                                            ((sample_index * in_channels + input_channel) * input_height + input_row) * input_width + input_col;
                                        // (filter, channel, kernel-row, kernel-col)
                                        // flat index into the filters.
                                        size_t weight_index =
                                            ((filter_index * in_channels + input_channel) * kernel_size + kernel_row) * kernel_size + kernel_col;

                                        window_sum += cached_input_[input_index] * weights[weight_index];
                                    }
                                }
                            }
                        }

                        // Store the finished output pixel (NCHW layout).
                        size_t output_index =
                            ((sample_index * out_channels + filter_index) * output_height + output_row) * output_width + output_col;
                        output_maps[output_index] = window_sum;
                    }
                }
            }
        }

        return output_maps;
    }

    // Propagates loss gradients back to weights, bias, and the input.
    // Takes output_gradient (dLoss/dOutput, same shape as forward's output).
    // Accumulates dLoss/dWeight and dLoss/dBias into grad_weights/grad_bias
    // (summed over batch and spatial positions = correct for mean loss since the loss already divides by batch size upstream).
    // Returns dLoss/dInput with the cached input's shape for earlier layers.
    Tensor backward(const Tensor& output_gradient) override {
        // Geometry comes from the cached forward input and the incoming grad.
        size_t batch_size = cached_input_.shape[0];
        size_t input_height = cached_input_.shape[2];
        size_t input_width = cached_input_.shape[3];

        size_t output_height = output_gradient.shape[2];
        size_t output_width = output_gradient.shape[3];

        // Fresh gradient buffers for this batch.
        grad_weights.data.assign(grad_weights.size(), 0.0f);
        grad_bias.data.assign(grad_bias.size(), 0.0f);

        // Gradient flowing to the previous layer (same shape as the input).
        Tensor input_gradient(cached_input_.shape, 0.0f);

        // Weight/bias grads are shared across the batch -> per-thread locals
        // merged at the end. Input-grad tiles are disjoint per sample, so they
        // need no locking (each thread owns its samples).
        #ifdef _OPENMP
        #pragma omp parallel
        {
            // Private accumulators sized like the shared gradient buffers.
            std::vector<float> local_weight_grad(grad_weights.size(), 0.0f);
            std::vector<float> local_bias_grad(grad_bias.size(), 0.0f);
            #pragma omp for schedule(static) nowait
            for (size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
                for (size_t filter_index = 0; filter_index < out_channels; ++filter_index) {
                    for (size_t output_row = 0; output_row < output_height; ++output_row) {
                        for (size_t output_col = 0; output_col < output_width; ++output_col) {

                            // Upstream gradient at this output position.
                            size_t output_index =
                                ((sample_index * out_channels + filter_index) * output_height + output_row) * output_width + output_col;
                            float upstream_grad = output_gradient[output_index];

                            // dLoss/dBias: every position using this filter adds its grad.
                            local_bias_grad[filter_index] += upstream_grad;

                            for (size_t input_channel = 0; input_channel < in_channels; ++input_channel) {
                                for (size_t kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
                                    for (size_t kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {

                                        int input_row = static_cast<int>(output_row * stride + kernel_row) - static_cast<int>(padding);
                                        int input_col = static_cast<int>(output_col * stride + kernel_col) - static_cast<int>(padding);

                                        if (input_row >= 0 && input_row < static_cast<int>(input_height) &&
                                            input_col >= 0 && input_col < static_cast<int>(input_width)) {

                                            size_t input_index =
                                                ((sample_index * in_channels + input_channel) * input_height + input_row) * input_width + input_col;
                                            size_t weight_index =
                                                ((filter_index * in_channels + input_channel) * kernel_size + kernel_row) * kernel_size + kernel_col;

                                            // dLoss/dWeight += input * upstream; dLoss/dInput += weight * upstream.
                                            local_weight_grad[weight_index] += cached_input_[input_index] * upstream_grad;
                                            input_gradient[input_index] += weights[weight_index] * upstream_grad;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Merge every thread's private sums into the shared buffers once.
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
        // Single-threaded reference path: identical math, direct accumulation.
        for (size_t sample_index = 0; sample_index < batch_size; ++sample_index) {
            for (size_t filter_index = 0; filter_index < out_channels; ++filter_index) {
                for (size_t output_row = 0; output_row < output_height; ++output_row) {
                    for (size_t output_col = 0; output_col < output_width; ++output_col) {

                        size_t output_index =
                            ((sample_index * out_channels + filter_index) * output_height + output_row) * output_width + output_col;
                        float upstream_grad = output_gradient[output_index];

                        grad_bias[filter_index] += upstream_grad;

                        for (size_t input_channel = 0; input_channel < in_channels; ++input_channel) {
                            for (size_t kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
                                for (size_t kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {

                                    int input_row = static_cast<int>(output_row * stride + kernel_row) - static_cast<int>(padding);
                                    int input_col = static_cast<int>(output_col * stride + kernel_col) - static_cast<int>(padding);

                                    if (input_row >= 0 && input_row < static_cast<int>(input_height) &&
                                        input_col >= 0 && input_col < static_cast<int>(input_width)) {

                                        size_t input_index =
                                            ((sample_index * in_channels + input_channel) * input_height + input_row) * input_width + input_col;
                                        size_t weight_index =
                                            ((filter_index * in_channels + input_channel) * kernel_size + kernel_row) * kernel_size + kernel_col;

                                        // Gradients w.r.t weights and input.
                                        grad_weights[weight_index] += cached_input_[input_index] * upstream_grad;
                                        input_gradient[input_index] += weights[weight_index] * upstream_grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        #endif

        return input_gradient;
    }

private:
    // Copy of the forward input batch, kept for forming gradients in backward().
    Tensor cached_input_;
};

} // namespace neuracpp

#endif // CONV2D_HPP
