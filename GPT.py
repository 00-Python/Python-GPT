# Import the numpy library as np for numerical operations
import numpy as np

# Define a class named GPT
class GPT:
    # Constructor of the class with parameters for input, hidden, and output dimensions
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialize class variables with the provided dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights with small random values
        self.Wxh = np.random.randn(hidden_dim, input_dim) * 0.01  # Weights for input to hidden layer
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01  # Weights for hidden to hidden layer (recurrent)
        self.Why = np.random.randn(output_dim, hidden_dim) * 0.01  # Weights for hidden to output layer

        # Initialize biases with zeros
        self.bh = np.zeros((hidden_dim, 1))  # Bias for hidden layer
        self.by = np.zeros((output_dim, 1))  # Bias for output layer

    # Forward pass through the network
    def forward(self, inputs):
        # Initialize previous hidden state as a vector of zeros
        h_prev = np.zeros((self.hidden_dim, 1))

        # Dictionaries to store the inputs, hidden states, and outputs at each time step
        xs, hs, ys = {}, {}, {}

        # Loop through each time step in the input sequence
        for t in range(len(inputs)):
            # One-hot encode the input at time step t
            xs[t] = np.zeros((self.input_dim, 1))
            xs[t][inputs[t]] = 1

            # Compute the hidden state at time step t
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, h_prev) + self.bh)

            # Compute the output at time step t
            ys[t] = np.dot(self.Why, hs[t]) + self.by

            # Update the previous hidden state to the current one for the next time step
            h_prev = hs[t]

        # Return the sequence of inputs, hidden states, and outputs
        return xs, hs, ys

    # Backward pass through the network
    def backward(self, xs, hs, ys, targets):
        # Initialize gradients as zero arrays with the same shape as the weights/biases
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    
        # Initialize the next hidden state gradient as a zero vector
        dhnext = np.zeros_like(hs[0])
    
        # Loop backwards through time steps
        for t in reversed(range(len(targets))):
            # Calculate the output gradient as the difference between predicted and target values
            dy = np.copy(ys[t])
            dy[targets[t]] -= 1
    
            # Calculate the gradients for the weights and biases connecting hidden to output
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
    
            # Backpropagate the gradient through the network
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - hs[t] * hs[t]) * dh  # Backprop through tanh nonlinearity
    
            # Calculate the gradients for the biases and weights
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T if t != 0 else 0)
    
            # Update the next hidden state gradient
            dhnext = np.dot(self.Whh.T, dhraw)
    
        # Return the gradients for weights and biases
        return dWxh, dWhh, dWhy, dbh, dby
        

    # Update the model's weights and biases using the gradients and learning rate
    def update(self, dWxh, dWhh, dWhy, dbh, dby, learning_rate):
        # Apply the updates to the weights and biases
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

    # Train the model on a given set of inputs and targets
    def train(self, input_indices, target_indices, learning_rate=0.1):
        # Perform forward pass
        xs, hs, ys = self.forward(input_indices)
        # Perform backward pass
        dWxh, dWhh, dWhy, dbh, dby = self.backward(xs, hs, ys, target_indices)
        # Update the weights and biases
        self.update(dWxh, dWhh, dWhy, dbh, dby, learning_rate)

    def predict(self, start_index, num_chars):
        # Initialize the input index
        input_index = start_index
    
        # Initialize previous hidden state as a vector of zeros
        h_prev = np.zeros((self.hidden_dim, 1))
    
        # List to store the predicted indices
        predicted_indices = []
    
        # Generate characters
        for _ in range(num_chars):
            # One-hot encode the input
            x = np.zeros((self.input_dim, 1))
            x[input_index] = 1
    
            # Forward pass through the network
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
            y = np.dot(self.Why, h) + self.by
    
            # Update the previous hidden state
            h_prev = h
    
            # Get the index of the highest score in the output layer as the next character
            input_index = np.argmax(y)
            predicted_indices.append(input_index)
    
        return predicted_indices
    


if __name__ == '__main__':
    # Define a simple text corpus
    text = "hello world"

    # Create a mapping of unique characters to integers
    chars = list(set(text))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    # Convert the text into a sequence of integer indices
    input_indices = [char_to_ix[ch] for ch in text]
    target_indices = input_indices[1:] + [input_indices[0]]

    # Initialize the GPT model
    model = GPT(input_dim=len(chars), hidden_dim=20, output_dim=len(chars))

    # Train the model on the input and target indices
    for epoch in range(1000):
        model.train(input_indices, target_indices, learning_rate=0.1)

    start_char = 'h'
    num_chars_to_predict = 500  # Specify the number of characters you want to predict

    # Get the index of the start character
    start_index = char_to_ix[start_char]

    # Predict the sequence of characters
    predicted_indices = model.predict(start_index, num_chars_to_predict)

    # Convert the predicted indices to characters
    predicted_sequence = ''.join(ix_to_char[idx] for idx in predicted_indices)

    print(f"Predicted sequence: {predicted_sequence}")
