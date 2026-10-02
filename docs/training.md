# Training driver (`src/main.cpp`)

Loads MNIST, trains `DigitNetwork` with plain SGD, evaluates each epoch,
optionally saves weights. Four visible phases mirror the console output.

## The network (`DigitNetwork`)

```
(B,1,28,28) → Conv(1→32,3×3,s1,p1) → ReLU → Pool(2) → (B,32,14,14)
             → Conv(32→64,3×3,s1,p1) → ReLU → Pool(2) → (B,64,7,7)
             → Flatten → (B,3136) → Linear(3136→128) → ReLU → Linear(128→10)
```

`forward()` chains the nine stages; `backward()` unwinds them in reverse,
filling each layer's gradient buffers. ~421k parameters total, ~99.5% of them
in `hidden_fc` (3136×128).

## The four phases per batch

```
1. optimizer.zero_grad()              — clear last batch's gradients
2. logits = network.forward(images)   — logits: (batch, 10)
   loss = CrossEntropyLoss.forward    — mean -log(prob[target])
3. network.backward(loss.backward())  — (prob − one_hot)/B flows back
4. optimizer.step()                   — W -= lr * grad on all 8 tensors
```

Accuracy counts `argmax(logits) == label` rows. A heartbeat line every 100
batches (`batch 100/937 | avg loss so far`) keeps long epochs transparent.

## CLI

```
--epochs N  --batch-size N  --lr F  --output PATH  --load PATH
--train-images P  --train-labels P  --test-images P  --test-labels P
--no-eval  --help                          (plus legacy "neuracpp 5 64 0.01")
```

`--output ""` skips saving; `--load` warm-starts from a checkpoint. Every run
also writes `checkpoints/history.csv`
(`epoch,train_loss,train_acc,test_loss,test_acc,seconds`, test columns -1
when `--no-eval` skips evaluation) for `python/plot_results.py`.

## Worked loss example (batch of 2)

```
logits row 0 → probs [0.7, 0.2, 0.1], target 0 → loss −log(0.7) = 0.357
logits row 1 → probs [0.1, 0.8, 0.1], target 1 → loss −log(0.8) = 0.223
batch loss = (0.357 + 0.223) / 2 = 0.290, grads = (prob − one_hot)/2
```
