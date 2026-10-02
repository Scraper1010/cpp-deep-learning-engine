#ifndef MODEL_IO_HPP
#define MODEL_IO_HPP

// Checkpoint file for the NeuraCPP deep network:
//   Conv2D(1->32, 3x3) -> ReLU -> MaxPool(2)
//   Conv2D(32->64, 3x3) -> ReLU -> MaxPool(2)
//   Flatten(3136) -> Linear(3136->128) -> ReLU -> Linear(128->10)
//
// Binary layout, little-endian, float32 values:
//   bytes  0-7 : identification tag "NEURACPP" (8 ASCII bytes)
//   next 14 x u64 : network shape, in order:
//       conv1_in, conv1_out, conv1_kernel, conv1_stride, conv1_padding,
//       conv2_in, conv2_out, conv2_kernel, conv2_stride, conv2_padding,
//       fc1_in, fc1_out, fc2_in, fc2_out
//   then 8 tensors, each as (u64 element_count + that many float32 values):
//       conv1 weights, conv1 bias, conv2 weights, conv2 bias,
//       fc1 weights, fc1 bias, fc2 weights, fc2 bias
//
// The loader checks the tag and the shape, then fills the live layers.
// Only checkpoints written by this exact network layout can be loaded.
// The Python twin reads the identical layout: see python/neuracpp_predict.py.
// Details and a byte-level example: docs/model_io.md.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "conv2d.hpp"
#include "linear.hpp"

namespace neuracpp {

namespace detail {

// Identification tag every checkpoint starts with ("NEURACPP", 8 bytes).
inline const char checkpoint_tag[8] = {'N', 'E', 'U', 'R', 'A', 'C', 'P', 'P'};

// Appends one little-endian u64 shape value to the open binary stream.
inline void write_u64(std::ofstream& output_stream, uint64_t shape_value) {
    output_stream.write(reinterpret_cast<const char*>(&shape_value), sizeof(shape_value));
}

// Appends one tensor as (u64 count + raw float32 bytes) to the stream.
inline void write_f32_vec(std::ofstream& output_stream, const std::vector<float>& values) {
    write_u64(output_stream, static_cast<uint64_t>(values.size()));
    if (!values.empty()) {
        output_stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    }
}

// Reads one u64 shape value, throwing if the file ends early.
inline uint64_t read_u64(std::ifstream& input_stream) {
    uint64_t shape_value = 0;
    input_stream.read(reinterpret_cast<char*>(&shape_value), sizeof(shape_value));
    if (!input_stream) throw std::runtime_error("Checkpoint ended early while reading its shape header.");
    return shape_value;
}

// Reads one (u64 count + float32 bytes) tensor, throwing on truncation.
inline std::vector<float> read_f32_vec(std::ifstream& input_stream, const char* tensor_name) {
    uint64_t element_count = read_u64(input_stream);
    if (element_count > (1ULL << 30)) {
        throw std::runtime_error(std::string("Checkpoint holds an impossible size for ") + tensor_name);
    }
    std::vector<float> values(static_cast<size_t>(element_count));
    if (element_count) {
        input_stream.read(reinterpret_cast<char*>(values.data()), element_count * sizeof(float));
        if (!input_stream) {
            throw std::runtime_error(std::string("Checkpoint ended early inside ") + tensor_name);
        }
    }
    return values;
}

// Verifies the 8-byte identification tag at the file start.
inline void check_tag(std::ifstream& input_stream, const std::string& file_path) {
    char tag_buffer[8];
    input_stream.read(tag_buffer, 8);
    if (!input_stream) throw std::runtime_error("Checkpoint is too short to hold its tag: " + file_path);
    for (int tag_index = 0; tag_index < 8; ++tag_index) {
        if (tag_buffer[tag_index] != checkpoint_tag[tag_index]) {
            throw std::runtime_error("Not a NeuraCPP checkpoint (bad tag): " + file_path);
        }
    }
}

} // namespace detail

// Writes the four layers' weights and biases to a single checkpoint file.
// Takes the destination path plus the two convolutions and two linear layers.
// Creates missing parent folders (so "checkpoints/model.bin" works fresh).
// Throws std::runtime_error if the file cannot be written.
inline void save_model(const std::string& file_path,
                       const Conv2D& first_conv, const Conv2D& second_conv,
                       const Linear& first_fc, const Linear& second_fc) {
    // Create e.g. "checkpoints/" automatically on a fresh machine.
    if (auto parent_folder = std::filesystem::path(file_path).parent_path(); !parent_folder.empty()) {
        std::error_code folder_error;
        std::filesystem::create_directories(parent_folder, folder_error);
        if (folder_error) throw std::runtime_error("Cannot create folder for: " + file_path);
    }
    std::ofstream output_stream(file_path, std::ios::binary);
    if (!output_stream.is_open()) throw std::runtime_error("Cannot open for writing: " + file_path);

    // Tag + network shape header.
    output_stream.write(detail::checkpoint_tag, 8);
    detail::write_u64(output_stream, first_conv.in_channels);
    detail::write_u64(output_stream, first_conv.out_channels);
    detail::write_u64(output_stream, first_conv.kernel_size);
    detail::write_u64(output_stream, first_conv.stride);
    detail::write_u64(output_stream, first_conv.padding);
    detail::write_u64(output_stream, second_conv.in_channels);
    detail::write_u64(output_stream, second_conv.out_channels);
    detail::write_u64(output_stream, second_conv.kernel_size);
    detail::write_u64(output_stream, second_conv.stride);
    detail::write_u64(output_stream, second_conv.padding);
    detail::write_u64(output_stream, first_fc.in_features);
    detail::write_u64(output_stream, first_fc.out_features);
    detail::write_u64(output_stream, second_fc.in_features);
    detail::write_u64(output_stream, second_fc.out_features);

    // The eight parameter tensors, in fixed order.
    detail::write_f32_vec(output_stream, first_conv.weights.data);
    detail::write_f32_vec(output_stream, first_conv.bias.data);
    detail::write_f32_vec(output_stream, second_conv.weights.data);
    detail::write_f32_vec(output_stream, second_conv.bias.data);
    detail::write_f32_vec(output_stream, first_fc.weights.data);
    detail::write_f32_vec(output_stream, first_fc.bias.data);
    detail::write_f32_vec(output_stream, second_fc.weights.data);
    detail::write_f32_vec(output_stream, second_fc.bias.data);

    output_stream.flush();
    if (!output_stream) throw std::runtime_error("Failed while writing: " + file_path);
}

// Loads a checkpoint into the four live layers, checking shape compatibility.
// Takes the file path plus mutable references to the network's layers.
// Throws std::runtime_error on missing files, bad tags, shape mismatches,
// or truncated data (a mismatched file usually means it was trained by a
// different network layout and the model must be retrained).
inline void load_model(const std::string& file_path,
                       Conv2D& first_conv, Conv2D& second_conv,
                       Linear& first_fc, Linear& second_fc) {
    std::ifstream input_stream(file_path, std::ios::binary);
    if (!input_stream.is_open()) throw std::runtime_error("Cannot open for reading: " + file_path);

    // The file must start with the identification tag.
    detail::check_tag(input_stream, file_path);

    // Read the 14 stored shape values in the documented order.
    uint64_t stored_shape[14];
    for (int shape_index = 0; shape_index < 14; ++shape_index) {
        stored_shape[shape_index] = detail::read_u64(input_stream);
    }

    // The stored network layout must match the live layers exactly.
    if (stored_shape[0] != first_conv.in_channels || stored_shape[1] != first_conv.out_channels ||
        stored_shape[2] != first_conv.kernel_size || stored_shape[3] != first_conv.stride ||
        stored_shape[4] != first_conv.padding ||
        stored_shape[5] != second_conv.in_channels || stored_shape[6] != second_conv.out_channels ||
        stored_shape[7] != second_conv.kernel_size || stored_shape[8] != second_conv.stride ||
        stored_shape[9] != second_conv.padding ||
        stored_shape[10] != first_fc.in_features || stored_shape[11] != first_fc.out_features ||
        stored_shape[12] != second_fc.in_features || stored_shape[13] != second_fc.out_features) {
        throw std::runtime_error("Checkpoint was written for a different network layout; please retrain it.");
    }

    // Read the eight tensors in the same order save_model wrote them.
    auto conv1_weights = detail::read_f32_vec(input_stream, "first convolution weights");
    auto conv1_bias = detail::read_f32_vec(input_stream, "first convolution bias");
    auto conv2_weights = detail::read_f32_vec(input_stream, "second convolution weights");
    auto conv2_bias = detail::read_f32_vec(input_stream, "second convolution bias");
    auto fc1_weights = detail::read_f32_vec(input_stream, "first linear weights");
    auto fc1_bias = detail::read_f32_vec(input_stream, "first linear bias");
    auto fc2_weights = detail::read_f32_vec(input_stream, "second linear weights");
    auto fc2_bias = detail::read_f32_vec(input_stream, "second linear bias");

    // Blob sizes must equal the live buffers exactly.
    if (conv1_weights.size() != first_conv.weights.size() || conv1_bias.size() != first_conv.bias.size() ||
        conv2_weights.size() != second_conv.weights.size() || conv2_bias.size() != second_conv.bias.size() ||
        fc1_weights.size() != first_fc.weights.size() || fc1_bias.size() != first_fc.bias.size() ||
        fc2_weights.size() != second_fc.weights.size() || fc2_bias.size() != second_fc.bias.size()) {
        throw std::runtime_error("Checkpoint tensor sizes do not match the live network.");
    }

    // Install the loaded values into the live layers.
    first_conv.weights.data = std::move(conv1_weights);
    first_conv.bias.data = std::move(conv1_bias);
    second_conv.weights.data = std::move(conv2_weights);
    second_conv.bias.data = std::move(conv2_bias);
    first_fc.weights.data = std::move(fc1_weights);
    first_fc.bias.data = std::move(fc1_bias);
    second_fc.weights.data = std::move(fc2_weights);
    second_fc.bias.data = std::move(fc2_bias);
}

} // namespace neuracpp

#endif // MODEL_IO_HPP
