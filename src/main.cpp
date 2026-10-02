// NeuraCPP training driver: loads MNIST, trains the deep CNN with plain SGD,
// evaluates on the test split each epoch, and optionally saves the weights.
//
// Network (see docs/training.md for the full walkthrough):
//   Conv(1->32, 3x3, stride 1, pad 1) -> ReLU -> MaxPool(2x2 stride 2)
//   Conv(32->64, 3x3, stride 1, pad 1) -> ReLU -> MaxPool(2x2 stride 2)
//   Flatten(64*7*7 = 3136) -> Linear(3136->128) -> ReLU -> Linear(128->10)
//   CrossEntropyLoss, vanilla SGD.
//
// Command line:
//   neuracpp --epochs 5 --output checkpoints/model.bin
//   neuracpp --epochs 10 --batch-size 128 --lr 0.01 --output model.bin --load model.bin
//   neuracpp --help

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <string>

#include "tensor.hpp"
#include "conv2d.hpp"
#include "relu.hpp"
#include "maxpool2d.hpp"
#include "flatten.hpp"
#include "linear.hpp"
#include "cross_entropy.hpp"
#include "sgd.hpp"
#include "mnist_reader.hpp"
#include "dataloader.hpp"
#include "model_io.hpp"

using namespace neuracpp;

// Returns the predicted digit (0-9) for one row of a logit batch.
// Takes the (batch, classes) logit tensor, the row to inspect, and the class count.
// Scans the row for its maximum value and returns that column's index.
static int argmax_prediction(const Tensor& logit_batch, size_t sample_index, size_t class_count) {
    // Assume class 0 wins until a larger logit is found.
    int best_class = 0;
    float best_logit = logit_batch.data[sample_index * class_count + 0];
    // Compare every remaining class in the row.
    for (size_t class_index = 1; class_index < class_count; ++class_index) {
        float candidate_logit = logit_batch.data[sample_index * class_count + class_index];
        if (candidate_logit > best_logit) {
            best_logit = candidate_logit;
            best_class = static_cast<int>(class_index);
        }
    }
    return best_class;
}

// The trainable network as one unit: owns every layer and chains them.
// Forward runs the full pipeline; backward unwinds it in reverse order.
// Layer order and shapes are documented on each member below.
struct DigitNetwork {
    // Block 1: (batch, 1, 28, 28) -> (batch, 32, 28, 28) -> ReLU -> pool to (batch, 32, 14, 14).
    Conv2D first_conv{1, 32, 3, 1, 1};
    // Rectifier after the first convolution (adds non-linearity).
    ReLU first_activation;
    // Halves each spatial axis: 28x28 maps become 14x14.
    MaxPool2D first_pool{2, 2};
    // Block 2: (batch, 32, 14, 14) -> (batch, 64, 14, 14) -> ReLU -> pool to (batch, 64, 7, 7).
    Conv2D second_conv{32, 64, 3, 1, 1};
    // Rectifier after the second convolution.
    ReLU second_activation;
    // Halves again: 14x14 maps become 7x7.
    MaxPool2D second_pool{2, 2};
    // Collapses (batch, 64, 7, 7) into (batch, 3136) rows for the classifier.
    Flatten flattener;
    // Hidden classifier: 3136 features -> 128 activations.
    Linear hidden_fc{64 * 7 * 7, 128};
    // Rectifier between the two classifier layers.
    ReLU hidden_activation;
    // Output classifier: 128 hidden values -> 10 digit logits.
    Linear output_fc{128, 10};

    // Runs an image batch through every stage to produce digit logits.
    // Takes input_images with shape (batch, 1, 28, 28).
    // Returns logits with shape (batch, 10); caches inside each layer for backward().
    Tensor forward(const Tensor& input_images) {
        Tensor hidden = first_conv.forward(input_images);
        hidden = first_activation.forward(hidden);
        hidden = first_pool.forward(hidden);
        hidden = second_conv.forward(hidden);
        hidden = second_activation.forward(hidden);
        hidden = second_pool.forward(hidden);
        hidden = flattener.forward(hidden);
        hidden = hidden_fc.forward(hidden);
        hidden = hidden_activation.forward(hidden);
        hidden = output_fc.forward(hidden);
        return hidden;
    }

    // Propagates a loss gradient back through every stage in reverse.
    // Takes the loss gradient over the logits, shape (batch, 10).
    // Fills each layer's grad_weights/grad_bias along the way.
    // Returns the gradient over the input images for completeness.
    Tensor backward(const Tensor& logit_gradient) {
        Tensor grad = output_fc.backward(logit_gradient);
        grad = hidden_activation.backward(grad);
        grad = hidden_fc.backward(grad);
        grad = flattener.backward(grad);
        grad = second_pool.backward(grad);
        grad = second_activation.backward(grad);
        grad = second_conv.backward(grad);
        grad = first_pool.backward(grad);
        grad = first_activation.backward(grad);
        grad = first_conv.backward(grad);
        return grad;
    }
};

// Prints the command-line help text describing every flag with an example.
static void print_usage(const char* program_name) {
    std::cout << "Usage:\n"
              << "  " << program_name << " [options]\n\n"
              << "Deep network (plain SGD, no batch-norm, no dropout):\n"
              << "  Conv(1->32,3x3) -> ReLU -> MaxPool2\n"
              << "  Conv(32->64,3x3) -> ReLU -> MaxPool2\n"
              << "  Flatten(3136) -> Linear(3136->128) -> ReLU -> Linear(128->10)\n\n"
              << "Options:\n"
              << "  --epochs N          Training epochs (default 5)\n"
              << "  --batch-size N      Mini-batch size (default 64)\n"
              << "  --lr F              Learning rate (default 0.01)\n"
              << "  --output PATH       Save trained weights (e.g. --output checkpoints/model.bin)\n"
              << "                      Empty string \"\" means don't save.\n"
              << "  --load PATH         Load weights before training (resume / eval)\n"
              << "  --train-images P    Path to train-images-idx3-ubyte\n"
              << "  --train-labels P    Path to train-labels-idx1-ubyte\n"
              << "  --test-images P     Path to t10k-images-idx3-ubyte\n"
              << "  --test-labels P     Path to t10k-labels-idx1-ubyte\n"
              << "  --no-eval           Skip test-set evaluation\n"
              << "  -h, --help          Show this help\n\n"
              << "Examples:\n"
              << "  " << program_name << " --epochs 5 --output checkpoints/model.bin\n"
              << "  " << program_name << " --epochs 10 --batch-size 128 --lr 0.01 --output model.bin\n"
              << "Legacy positional form (still works):\n"
              << "  " << program_name << " [epochs] [batch_size] [lr]\n";
}

int main(int argument_count, char* argument_values[]) {
    std::cout << "========================================\n";
    std::cout << "   NeuraCPP: High-Performance Engine    \n";
    std::cout << "========================================\n\n";

    // Default training settings, overridden by the flags parsed below.
    size_t batch_size = 64;          // Rows per gradient step.
    size_t epoch_count = 5;          // Full passes over the training set.
    float learning_rate = 0.01f;    // SGD step size.
    std::string output_path;        // Where to save weights (empty = skip saving).
    std::string load_path;          // Checkpoint to start from (empty = random init).

    // Default dataset file locations (created by the download scripts).
    std::string train_images_path = "data/train-images-idx3-ubyte";
    std::string train_labels_path = "data/train-labels-idx1-ubyte";
    std::string test_images_path = "data/t10k-images-idx3-ubyte";
    std::string test_labels_path = "data/t10k-labels-idx1-ubyte";
    // When false, the test split is neither loaded nor evaluated.
    bool run_evaluation = true;

    // Reads the value after a space-separated flag, e.g. "--epochs 5".
    // Exits with code 2 if the flag is the last argument (nothing follows it).
    auto read_flag_value = [&](int flag_position, const std::string& flag_name) -> std::string {
        if (flag_position + 1 >= argument_count) {
            std::cerr << "Error: " << flag_name << " needs a value.\n";
            print_usage(argument_values[0]);
            std::exit(2);
        }
        return argument_values[flag_position + 1];
    };
    // Splits "--flag=value" into ("--flag", "value"); no '=' returns ("--flag", "").
    auto split_equals_form = [](const std::string& raw_argument) -> std::pair<std::string, std::string> {
        size_t equals_at = raw_argument.find('=');
        if (equals_at == std::string::npos) return {raw_argument, ""};
        return {raw_argument.substr(0, equals_at), raw_argument.substr(equals_at + 1)};
    };

    // Positional arguments kept for the legacy "neuracpp 5 64 0.01" form.
    std::vector<std::string> positional_args;
    for (int arg_index = 1; arg_index < argument_count; ++arg_index) {
        std::string current_arg = argument_values[arg_index];
        if (current_arg == "-h" || current_arg == "--help") { print_usage(argument_values[0]); return 0; }
        else if (current_arg == "--epochs" || current_arg.rfind("--epochs=", 0) == 0) {
            std::string value_text = (current_arg == "--epochs") ? read_flag_value(arg_index, current_arg) : split_equals_form(current_arg).second;
            if (current_arg == "--epochs") ++arg_index;
            epoch_count = static_cast<size_t>(std::stoul(value_text));
        } else if (current_arg == "--batch-size" || current_arg.rfind("--batch-size=", 0) == 0 ||
                   current_arg == "--batch_size" || current_arg.rfind("--batch_size=", 0) == 0) {
            std::string value_text = (current_arg.find('=') == std::string::npos) ? read_flag_value(arg_index, current_arg) : split_equals_form(current_arg).second;
            if (current_arg.find('=') == std::string::npos) ++arg_index;
            batch_size = static_cast<size_t>(std::stoul(value_text));
        } else if (current_arg == "--lr" || current_arg.rfind("--lr=", 0) == 0) {
            std::string value_text = (current_arg == "--lr") ? read_flag_value(arg_index, current_arg) : split_equals_form(current_arg).second;
            if (current_arg == "--lr") ++arg_index;
            learning_rate = std::stof(value_text);
        } else if (current_arg == "--output" || current_arg.rfind("--output=", 0) == 0) {
            std::string value_text = (current_arg == "--output") ? read_flag_value(arg_index, current_arg) : split_equals_form(current_arg).second;
            if (current_arg == "--output") ++arg_index;
            output_path = value_text; // "" is a valid no-op meaning "don't save".
        } else if (current_arg == "--load" || current_arg.rfind("--load=", 0) == 0) {
            std::string value_text = (current_arg == "--load") ? read_flag_value(arg_index, current_arg) : split_equals_form(current_arg).second;
            if (current_arg == "--load") ++arg_index;
            load_path = value_text;
        } else if (current_arg == "--train-images" || current_arg.rfind("--train-images=", 0) == 0) {
            std::string value_text = (current_arg.find('=') == std::string::npos) ? read_flag_value(arg_index, current_arg) : split_equals_form(current_arg).second;
            if (current_arg.find('=') == std::string::npos) ++arg_index;
            train_images_path = value_text;
        } else if (current_arg == "--train-labels" || current_arg.rfind("--train-labels=", 0) == 0) {
            std::string value_text = (current_arg.find('=') == std::string::npos) ? read_flag_value(arg_index, current_arg) : split_equals_form(current_arg).second;
            if (current_arg.find('=') == std::string::npos) ++arg_index;
            train_labels_path = value_text;
        } else if (current_arg == "--test-images" || current_arg.rfind("--test-images=", 0) == 0) {
            std::string value_text = (current_arg.find('=') == std::string::npos) ? read_flag_value(arg_index, current_arg) : split_equals_form(current_arg).second;
            if (current_arg.find('=') == std::string::npos) ++arg_index;
            test_images_path = value_text;
        } else if (current_arg == "--test-labels" || current_arg.rfind("--test-labels=", 0) == 0) {
            std::string value_text = (current_arg.find('=') == std::string::npos) ? read_flag_value(arg_index, current_arg) : split_equals_form(current_arg).second;
            if (current_arg.find('=') == std::string::npos) ++arg_index;
            test_labels_path = value_text;
        } else if (current_arg == "--no-eval") {
            run_evaluation = false;
        } else if (!current_arg.empty() && current_arg[0] == '-') {
            std::cerr << "Error: unknown flag " << current_arg << "\n";
            print_usage(argument_values[0]);
            return 2;
        } else {
            positional_args.push_back(current_arg);
        }
    }
    // Legacy form: neuracpp [epochs] [batch_size] [lr] [train_images_path].
    if (!positional_args.empty()) {
        try {
            if (positional_args.size() > 0) epoch_count = static_cast<size_t>(std::stoul(positional_args[0]));
            if (positional_args.size() > 1) batch_size = static_cast<size_t>(std::stoul(positional_args[1]));
            if (positional_args.size() > 2) learning_rate = std::stof(positional_args[2]);
            if (positional_args.size() > 3) train_images_path = positional_args[3];
        } catch (const std::exception& parse_error) {
            std::cerr << "Error parsing positional args: " << parse_error.what() << "\n";
            print_usage(argument_values[0]);
            return 2;
        }
    }
    // Degenerate settings would divide by zero or loop forever.
    if (batch_size == 0 || epoch_count == 0) {
        std::cerr << "Error: --epochs and --batch-size must be > 0.\n";
        return 2;
    }

    // Step 1: load the MNIST image/label archives into memory tensors.
    std::cout << "[1/4] Loading MNIST dataset...\n";
    MNISTData train_dataset;
    MNISTData test_dataset;
    bool test_available = false;
    try {
        train_dataset = MNISTReader::load(train_images_path, train_labels_path);
    } catch (const std::exception& load_error) {
        std::cerr << "Error: Failed to load training set: " << load_error.what() << "\n";
        std::cerr << "Hint: run ./download_mnist.sh (Linux/macOS) or ./download_mnist.ps1 (Windows) first.\n";
        return 1;
    }
    if (run_evaluation) {
        try {
            test_dataset = MNISTReader::load(test_images_path, test_labels_path);
            test_available = true;
        } catch (const std::exception& load_error) {
            std::cerr << "Warning: test set not found (" << load_error.what() << "). Skipping eval.\n";
        }
    }
    std::cout << "Loaded " << train_dataset.labels.size() << " training samples";
    if (test_available) std::cout << " + " << test_dataset.labels.size() << " test samples";
    std::cout << ".\n\n";

    // Step 2: build the network (~421k parameters, all He-initialized above).
    std::cout << "[2/4] Initializing Network Architecture...\n";
    std::cout << "  Conv(1->32,3x3) -> ReLU -> Pool2 | Conv(32->64,3x3) -> ReLU -> Pool2\n";
    std::cout << "  Flatten(3136) -> Linear(3136->128) -> ReLU -> Linear(128->10)\n";
    DigitNetwork network;
    CrossEntropyLoss loss_function;

    // Optional warm start: overwrite the random init with trained weights.
    if (!load_path.empty()) {
        std::cout << "Loading weights from " << load_path << " ...\n";
        try {
            load_model(load_path, network.first_conv, network.second_conv, network.hidden_fc, network.output_fc);
            std::cout << "Weights loaded.\n";
        } catch (const std::exception& load_error) {
            std::cerr << "Error loading model: " << load_error.what() << "\n";
            return 1;
        }
    }

    // Step 3: hand every learnable tensor to one SGD optimizer.
    std::cout << "[3/4] Registering parameters with SGD Optimizer (lr=" << learning_rate << ")...\n";
    SGD optimizer(learning_rate);
    optimizer.register_parameter(network.first_conv.weights.data, network.first_conv.grad_weights.data);
    optimizer.register_parameter(network.first_conv.bias.data, network.first_conv.grad_bias.data);
    optimizer.register_parameter(network.second_conv.weights.data, network.second_conv.grad_weights.data);
    optimizer.register_parameter(network.second_conv.bias.data, network.second_conv.grad_bias.data);
    optimizer.register_parameter(network.hidden_fc.weights.data, network.hidden_fc.grad_weights.data);
    optimizer.register_parameter(network.hidden_fc.bias.data, network.hidden_fc.grad_bias.data);
    optimizer.register_parameter(network.output_fc.weights.data, network.output_fc.grad_weights.data);
    optimizer.register_parameter(network.output_fc.bias.data, network.output_fc.grad_bias.data);

    // Batch samplers: shuffled each epoch for training, fixed order for testing.
    DataLoader train_loader(train_dataset, batch_size, true);
    DataLoader* test_loader_ptr = nullptr;
    DataLoader test_loader(test_dataset, 256, false);
    if (test_available) test_loader_ptr = &test_loader;

    // Step 4: the classic four-phase loop over epochs and mini-batches.
    std::cout << "\n[4/4] Starting Training Pipeline (" << epoch_count << " epochs)...\n";
    std::cout << "NOTE: the deep net is much bigger than a toy one - expect minutes per epoch on CPU.\n";
    std::cout << "---------------------------------------------------------\n";

    // Per-epoch metric history for the results plots (docs + README).
    // Written as CSV next to the checkpoint:
    // epoch, train_loss, train_acc, test_loss, test_acc, seconds.
    // Python (python/plot_results.py) renders training_curves.png + confusion_matrix.png from it.
    struct EpochRecord {
        size_t epoch;
        float train_loss;
        float train_acc;
        float test_loss;   // -1 when evaluation was skipped.
        float test_acc;    // -1 when evaluation was skipped.
        double seconds;
    };
    std::vector<EpochRecord> epoch_history;

    for (size_t epoch_index = 1; epoch_index <= epoch_count; ++epoch_index) {
        // Wall-clock timer for the epoch-time column.
        auto epoch_start = std::chrono::high_resolution_clock::now();
        // Fresh random order so batches differ every epoch.
        train_loader.shuffle_data();

        // Running totals for the epoch summary line.
        float summed_loss = 0.0f;
        size_t correct_count = 0;
        size_t seen_count = 0;
        size_t batch_total = train_loader.num_batches();

        for (size_t batch_index = 0; batch_index < batch_total; ++batch_index) {
            // Gather the next shuffled slice of images + labels.
            Batch training_batch = train_loader.get_batch(batch_index);
            size_t current_batch_size = training_batch.images.shape[0];

            // Phase 1: clear last batch's gradients so they never accumulate.
            optimizer.zero_grad();
            // Phase 2: forward pass + mean loss + running accuracy.
            Tensor batch_logits = network.forward(training_batch.images);
            float batch_loss = loss_function.forward(batch_logits, training_batch.labels);
            summed_loss += batch_loss * static_cast<float>(current_batch_size);

            // Count argmax hits for the training-accuracy column.
            for (size_t row_index = 0; row_index < current_batch_size; ++row_index) {
                if (argmax_prediction(batch_logits, row_index, 10) == training_batch.labels[row_index]) {
                    ++correct_count;
                }
            }
            seen_count += current_batch_size;

            // Phase 3: backpropagate the loss gradient into every parameter.
            Tensor loss_gradient = loss_function.backward();
            network.backward(loss_gradient);
            // Phase 4: apply one SGD step to all eight parameter tensors.
            optimizer.step();

            // Heartbeat line so long epochs never look frozen.
            if ((batch_index + 1) % 100 == 0 || (batch_index + 1) == batch_total) {
                float mean_loss_so_far = summed_loss / static_cast<float>(seen_count);
                std::cout << "  batch " << (batch_index + 1) << "/" << batch_total
                          << " | avg loss so far: " << std::fixed << std::setprecision(4) << mean_loss_so_far
                          << std::endl;
            }
        }

        // Epoch summary with timing.
        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_seconds = std::chrono::duration<double>(epoch_end - epoch_start).count();
        float train_accuracy = seen_count ? 100.0f * static_cast<float>(correct_count) / static_cast<float>(seen_count) : 0.0f;
        float mean_loss = seen_count ? summed_loss / static_cast<float>(seen_count) : 0.0f;

        std::cout << "Epoch " << epoch_index << "/" << epoch_count
                  << " | Loss: " << std::fixed << std::setprecision(4) << mean_loss
                  << " | Acc: " << std::fixed << std::setprecision(2) << train_accuracy << "%"
                  << " | Time: " << std::fixed << std::setprecision(1) << epoch_seconds << "s";

        // Forward-only pass over the test split (no gradients, no shuffling).
        // Accumulates the mean loss AND accuracy so the curves can show both.
        float test_accuracy = -1.0f;
        float test_mean_loss = -1.0f;
        if (test_available) {
            size_t test_correct = 0;
            size_t test_seen = 0;
            float test_summed_loss = 0.0f;
            CrossEntropyLoss eval_loss;  // Separate instance: forward() caches per call.
            for (size_t batch_index = 0; batch_index < test_loader_ptr->num_batches(); ++batch_index) {
                Batch eval_batch = test_loader_ptr->get_batch(batch_index);
                Tensor eval_logits = network.forward(eval_batch.images);
                test_summed_loss += eval_loss.forward(eval_logits, eval_batch.labels) *
                                    static_cast<float>(eval_batch.images.shape[0]);
                for (size_t row_index = 0; row_index < eval_batch.images.shape[0]; ++row_index) {
                    if (argmax_prediction(eval_logits, row_index, 10) == eval_batch.labels[row_index]) {
                        ++test_correct;
                    }
                }
                test_seen += eval_batch.images.shape[0];
            }
            test_mean_loss = test_seen ? test_summed_loss / static_cast<float>(test_seen) : -1.0f;
            test_accuracy = 100.0f * static_cast<float>(test_correct) / static_cast<float>(test_seen);
            std::cout << " | Test loss: " << std::fixed << std::setprecision(4) << test_mean_loss
                      << " | Test Acc: " << std::fixed << std::setprecision(2) << test_accuracy << "%";
        }
        std::cout << "\n";

        // Remember this epoch for the history file written after training.
        epoch_history.push_back({epoch_index, mean_loss, train_accuracy, test_mean_loss, test_accuracy, epoch_seconds});
    }

    std::cout << "---------------------------------------------------------\n";
    std::cout << "Training Complete!\n";

    // Persist the metric history (overwrites any previous run's file).
    {
        std::string history_path = "checkpoints/history.csv";
        if (auto parent_folder = std::filesystem::path(history_path).parent_path(); !parent_folder.empty()) {
            std::error_code folder_error;
            std::filesystem::create_directories(parent_folder, folder_error);
        }
        std::ofstream history_file(history_path);
        if (history_file.is_open()) {
            history_file << "epoch,train_loss,train_acc,test_loss,test_acc,seconds\n";
            for (const auto& record : epoch_history) {
                history_file << record.epoch << ","
                             << std::fixed << std::setprecision(4) << record.train_loss << ","
                             << std::fixed << std::setprecision(2) << record.train_acc << ","
                             << std::fixed << std::setprecision(4) << record.test_loss << ","
                             << std::fixed << std::setprecision(2) << record.test_acc << ","
                             << std::fixed << std::setprecision(1) << record.seconds << "\n";
            }
            std::cout << "Saved metrics to " << history_path << "\n";
        } else {
            std::cerr << "Warning: could not write " << history_path << "\n";
        }
    }

    // Persist the trained weights if the user asked for a checkpoint.
    if (!output_path.empty()) {
        try {
            save_model(output_path, network.first_conv, network.second_conv, network.hidden_fc, network.output_fc);
            std::cout << "Saved weights to " << output_path << "\n";
        } catch (const std::exception& save_error) {
            std::cerr << "Error saving model: " << save_error.what() << "\n";
            return 1;
        }
    }

    return 0;
}
