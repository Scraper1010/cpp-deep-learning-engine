# Main training loop (`src/main.cpp`)

The heart of the trainer: the epoch/batch loop where the four phases run.
(`DigitNetwork`, argument parsing, and checkpoint saving are covered in
[training.md](training.md) — this page is only the train code itself.)

## The code

```cpp
for (size_t epoch_index = 1; epoch_index <= epoch_count; ++epoch_index) {
    auto epoch_start = std::chrono::high_resolution_clock::now();
    train_loader.shuffle_data();   // fresh random batch order every epoch

    float summed_loss = 0.0f;      // accumulates mean-loss × batch size
    size_t correct_count = 0;      // argmax hits, for the accuracy column
    size_t seen_count = 0;         // samples processed this epoch
    size_t batch_total = train_loader.num_batches();

    for (size_t batch_index = 0; batch_index < batch_total; ++batch_index) {
        Batch training_batch = train_loader.get_batch(batch_index);
        size_t current_batch_size = training_batch.images.shape[0];

        // Phase 1: clear last batch's gradients so they never accumulate.
        optimizer.zero_grad();
        // Phase 2: forward pass + mean loss + running accuracy.
        Tensor batch_logits = network.forward(training_batch.images);
        float batch_loss = loss_function.forward(batch_logits, training_batch.labels);
        summed_loss += batch_loss * static_cast<float>(current_batch_size);

        for (size_t row_index = 0; row_index < current_batch_size; ++row_index) {
            if (argmax_prediction(batch_logits, row_index, 10)
                    == training_batch.labels[row_index]) {
                ++correct_count;
            }
        }
        seen_count += current_batch_size;

        // Phase 3: backpropagate the loss gradient into every parameter.
        Tensor loss_gradient = loss_function.backward();
        network.backward(loss_gradient);
        // Phase 4: apply one SGD step to all eight parameter tensors.
        optimizer.step();

        // Heartbeat so long epochs never look frozen.
        if ((batch_index + 1) % 100 == 0 || (batch_index + 1) == batch_total) {
            float mean_loss_so_far = summed_loss / static_cast<float>(seen_count);
            std::cout << "  batch " << (batch_index + 1) << "/" << batch_total
                      << " | avg loss so far: " << mean_loss_so_far << "\n";
        }
    }
    // ... epoch summary (loss / accuracy / timing / test eval) ...
}
```

## Reading it line by line

- **`train_loader.shuffle_data()`** — re-permutes the sample order, so every
  epoch slices different batches. Without it the network would see the same
  938 batches in the same order and memorize batch artifacts.
- **`get_batch(batch_index)`** — gathers the next shuffled slice into one
  contiguous `(current_batch_size, 1, 28, 28)` tensor plus labels. The final
  batch of the epoch is smaller (60000 = 937×64 + 32), hence
  `current_batch_size` instead of assuming 64.
- **Phase 1 `zero_grad()`** — every gradient buffer back to 0. `backward()`
  *adds* into gradients (`+=`), so skipping this would average this batch's
  gradients with every previous batch's.
- **Phase 2 forward + loss** — `batch_logits` is `(current_batch_size, 10)`;
  `loss_function.forward` returns the batch-*mean* loss, so it is re-weighted
  by `current_batch_size` into `summed_loss` (otherwise the small tail batch
  would count as much as a full one in the epoch mean).
- **Accuracy count** — `argmax_prediction` scans one logit row for its maximum
  column and compares it against the true digit label.
- **Phase 3 backward** — `loss_function.backward()` yields
  `(predicted_probability − target_one_hot) / batch_size`; `network.backward`
  carries it through `output_fc → hidden_activation → hidden_fc → flattener →
  second_pool → second_activation → second_conv → first_pool →
  first_activation → first_conv`, filling all eight gradient buffers.
- **Phase 4 `optimizer.step()`** — `parameter −= learning_rate × gradient`
  on every registered tensor (see [sgd.md](sgd.md)).
- **Heartbeat** — a progress line every 100 batches; with ~4-minute epochs the
  console would otherwise sit silent long enough to look hung.
