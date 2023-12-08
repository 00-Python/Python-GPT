# Python-GPT
This is a Python implementation of the GPT (Generative Pre-trained Transformer) model. The GPT model is a type of recurrent neural network that is trained to generate text data by predicting the next character or word in a sequence.

# Prerequisites

- Python 3.x
- Numpy

# Usage

To use the GPT model, follow these steps:

1. Import the numpy library:

```python
import numpy as np
```

2. Define a class named GPT:

```python
class GPT:
    def __init__(self, input_dim, hidden_dim, output_dim):
        ...
```

The `GPT` class has three parameters: `input_dim`, `hidden_dim`, and `output_dim`. These parameters define the dimensions of the input, hidden, and output layers of the GPT model.

3. Initialize the model with random weights and biases:

```python
self.Wxh = np.random.randn(hidden_dim, input_dim) * 0.01  # Weights for input to hidden layer
self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01  # Weights for hidden to hidden layer (recurrent)
self.Why = np.random.randn(output_dim, hidden_dim) * 0.01  # Weights for hidden to output layer

self.bh = np.zeros((hidden_dim, 1))  # Bias for hidden layer
self.by = np.zeros((output_dim, 1))  # Bias for output layer
```

4. Implement the forward pass of the model:

```python
def forward(self, inputs):
    ...
```

The `forward` method takes an array of input indices and returns the sequence of input, hidden state, and output values at each time step.

5. Implement the backward pass of the model:

```python
def backward(self, xs, hs, ys, targets):
    ...
```

The `backward` method takes the input, hidden state, output, and target sequences, and calculates the gradients for the weights and biases of the model.

6. Implement the update step to update the model's weights and biases:

```python
def update(self, dWxh, dWhh, dWhy, dbh, dby, learning_rate):
    ...
```

The `update` method applies the gradients to the weights and biases of the model using the given learning rate.

7. Train the model on a given set of inputs and targets:

```python
def train(self, input_indices, target_indices, learning_rate=0.1):
    ...
```

The `train` method performs the forward and backward pass, and updates the model's weights and biases on the given input and target sequences.

8. Generate predictions using the trained model:

```python
def predict(self, start_index, num_chars):
    ...
```

The `predict` method takes a start index and the number of characters to predict, and returns a sequence of predicted indices.

9. Test the model on a simple text corpus:

```python
if __name__ == '__main__':
    text = "hello world"

    chars = list(set(text))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    input_indices = [char_to_ix[ch] for ch in text]
    target_indices = input_indices[1:] + [input_indices[0]]

    model = GPT(input_dim=len(chars), hidden_dim=20, output_dim=len(chars))

    for epoch in range(1000):
        model.train(input_indices, target_indices, learning_rate=0.1)

    start_char = 'h'
    num_chars_to_predict = 500
    start_index = char_to_ix[start_char]

    predicted_indices = model.predict(start_index, num_chars_to_predict)

    predicted_sequence = ''.join(ix_to_char[idx] for idx in predicted_indices)

    print(f"Predicted sequence: {predicted_sequence}")
```

In this example, the model is trained on the input and target indices of the text "hello world" and then used to generate a sequence of characters starting with the letter 'h'.

# Dependencies

- Numpy - A library for numerical operations in Python. Install using `pip install numpy`.

