#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

// MNISTReader: parses the original MNIST IDX binary files into engine tensors.
// Image file (IDX3): u32 magic 2051, u32 image_count, u32 rows, u32 cols,
//   then image_count * rows * cols bytes (pixels 0-255, row-major).
// Label file (IDX1): u32 magic 2049, u32 label_count, then label_count bytes.
// All header integers are big-endian and byte-swapped on load. Pixels are
// scaled to [0, 1] floats; labels become int64 digits 0-9.
// See docs/mnist_reader.md for the byte layout and a worked header example.

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include "tensor.hpp"

namespace neuracpp {

// Images plus labels of one MNIST split (train or test).
struct MNISTData {
    // Grayscale images, shape (image_count, 1, 28, 28), pixels in [0.0, 1.0].
    Tensor images;
    // Digit per image (0-9), same length as the image batch dimension.
    std::vector<int64_t> labels;
};

class MNISTReader {
public:
    // Loads one MNIST split from its two IDX files.
    // Takes the image archive path (IDX3) and the label archive path (IDX1).
    // Throws std::runtime_error if files are missing, malformed, or mismatched.
    // Returns images normalized to [0, 1] plus integer labels.
    inline static MNISTData load(const std::string& images_path, const std::string& labels_path) {
        // -----------------------------------------------------------------
        // Part 1: read the IDX3 image archive.
        // -----------------------------------------------------------------
        std::ifstream image_file(images_path, std::ios::binary);
        if (!image_file.is_open()) {
            throw std::runtime_error("Failed to open image file: " + images_path);
        }

        // Four big-endian header words: magic, count, rows, columns.
        uint32_t file_magic = 0, image_count = 0, row_count = 0, col_count = 0;
        image_file.read(reinterpret_cast<char*>(&file_magic), 4);
        image_file.read(reinterpret_cast<char*>(&image_count), 4);
        image_file.read(reinterpret_cast<char*>(&row_count), 4);
        image_file.read(reinterpret_cast<char*>(&col_count), 4);

        // Convert the header from IDX big-endian to host byte order.
        file_magic = swap_endian(file_magic);
        image_count = swap_endian(image_count);
        row_count = swap_endian(row_count);
        col_count = swap_endian(col_count);

        // The MNIST image archive is identified by magic 2051.
        if (file_magic != 2051) {
            throw std::runtime_error("Invalid MNIST image magic number! Expected 2051, got " + std::to_string(file_magic));
        }

        // Read every raw pixel byte that follows the 16-byte header.
        size_t total_pixels = static_cast<size_t>(image_count) * row_count * col_count;
        std::vector<uint8_t> pixel_buffer(total_pixels);
        image_file.read(reinterpret_cast<char*>(pixel_buffer.data()), total_pixels);

        // Store as float images with shape (N, 1, rows, cols), scaled to [0, 1].
        Tensor images({static_cast<size_t>(image_count), 1, static_cast<size_t>(row_count), static_cast<size_t>(col_count)});
        for (size_t pixel_index = 0; pixel_index < total_pixels; ++pixel_index) {
            images[pixel_index] = static_cast<float>(pixel_buffer[pixel_index]) / 255.0f;
        }

        // -----------------------------------------------------------------
        // Part 2: read the IDX1 label archive.
        // -----------------------------------------------------------------
        std::ifstream label_file(labels_path, std::ios::binary);
        if (!label_file.is_open()) {
            throw std::runtime_error("Failed to open label file: " + labels_path);
        }

        // Two big-endian header words: magic and label count.
        uint32_t label_magic = 0, label_count = 0;
        label_file.read(reinterpret_cast<char*>(&label_magic), 4);
        label_file.read(reinterpret_cast<char*>(&label_count), 4);

        // Convert from IDX big-endian to host byte order.
        label_magic = swap_endian(label_magic);
        label_count = swap_endian(label_count);

        // The MNIST label archive is identified by magic 2049.
        if (label_magic != 2049) {
            throw std::runtime_error("Invalid MNIST label magic number! Expected 2049, got " + std::to_string(label_magic));
        }

        // Every image needs exactly one label.
        if (image_count != label_count) {
            throw std::runtime_error("Image count (" + std::to_string(image_count) +
                                     ") does not match label count (" + std::to_string(label_count) + ").");
        }

        // Read the raw label bytes that follow the 8-byte header.
        std::vector<uint8_t> label_buffer(label_count);
        label_file.read(reinterpret_cast<char*>(label_buffer.data()), label_count);

        // Widen each byte label to int64 for the engine's label type.
        std::vector<int64_t> labels(label_count);
        for (size_t label_index = 0; label_index < label_count; ++label_index) {
            labels[label_index] = static_cast<int64_t>(label_buffer[label_index]);
        }

        std::cout << "[MNISTReader] Loaded " << image_count << " images ("
                  << row_count << "x" << col_count << ") successfully.\n";

        return {std::move(images), std::move(labels)};
    }

private:
    // Reverses the four bytes of an IDX big-endian word into host order.
    // Takes the raw 32-bit word as read from disk.
    // Returns the byte-swapped value for comparison and sizing.
    inline static uint32_t swap_endian(uint32_t big_endian_value) {
        return ((big_endian_value << 24) & 0xFF000000) |
               ((big_endian_value << 8)  & 0x00FF0000) |
               ((big_endian_value >> 8)  & 0x0000FF00) |
               ((big_endian_value >> 24) & 0x000000FF);
    }
};

} // namespace neuracpp

#endif // MNIST_READER_HPP
