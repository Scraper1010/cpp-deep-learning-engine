# SGD (`include/sgd.hpp`)

Vanilla stochastic gradient descent: `parameter -= learning_rate × gradient`.

## Design

- `Parameter { data, grad }` — non-owning pointers into layer storage (e.g.
  `Linear::weights` + `Linear::grad_weights`), registered once.
- `register_parameter(data, grad)` — the trainer hands over all eight tensors
  (two convolutions × weights/bias, two linears × weights/bias).
- `zero_grad()` — fills every gradient buffer with 0 before a batch (gradients
  must never leak across batches).
- `step()` — applies `parameter_data[element_index] -= learning_rate *
  parameter_grad[element_index]` element-wise to all.

Typical batch: `zero_grad()` → forward → `backward()` (fills grads) → `step()`.

## Weights vs biases through the same code

A weight matrix and a bias vector look different — one is 2-D, one is 1-D —
but SGD never sees shapes, only flat buffers plus their gradients:

```
hidden layer: weights shape (3136, 128) → 401,408 floats
              bias    shape (1, 128)     →     128 floats

optimizer.register_parameter(hidden_fc.weights.data, hidden_fc.grad_weights.data);
optimizer.register_parameter(hidden_fc.bias.data,    hidden_fc.grad_bias.data);
```

Both registrations land in the same `registered_params` list as plain
`{data, grad}` pointer pairs. `step()` then runs one identical loop over each:

```
for each registered parameter:
    for each element_index in parameter_data:
        parameter_data[element_index] -= learning_rate * parameter_grad[element_index]
```

The 401,408 weight updates and the 128 bias updates are the same three lines —
no special-casing, because by update time every tensor is just "values next to
their gradients". Shapes mattered when *filling* the gradients (each layer's
`backward()` knows its own math); they no longer matter when *applying* them.

## Worked example (learning_rate = 0.1)

```
parameter_values = [1.0, 2.0],  parameter_gradients = [0.5, -1.0]
step → parameter_values = [1.0 - 0.1*0.5, 2.0 - 0.1*(-1.0)] = [0.95, 2.10]
```

No momentum, no schedules, no weight decay — deliberately the simplest
optimizer.
