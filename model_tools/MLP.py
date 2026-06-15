import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt


def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """calculate the binary cross entropy"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def plot_loss(training_loss):
    """plot the loss history"""
    plt.plot(training_loss)
    plt.title("Loss Function vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def plot_loss2(training_loss, validation_loss):
    """plot training and validation loss history"""
    plt.figure(figsize=(8, 5))

    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")

    plt.title("Loss Function vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.show()


class MLP:
    """multilayer perceptron class with 2 hidden layers"""
    def __init__(self, input_size, hidden_size, output_size, mu=1, sigma=1):
        """randomly initializes weights between 4 layers:
        input to hidden1,
        hidden1 to hidden2,
        hidden2 to output
        and sets biases to zero"""
        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size) * np.sqrt(1/input_size)
        self.weights_hidden1_hidden2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(1/hidden_size)
        self.weights_hidden2_output = np.random.randn(hidden_size, output_size) * np.sqrt(1/hidden_size)
        self.bias_hidden1 = np.zeros((1, hidden_size))
        self.bias_hidden2 = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.hidden_size = hidden_size
        self.mu = mu
        self.sigma = sigma

    def sigmoid(self, x):
        """sigmoid function"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        """forward propagation"""
        # X is the input
        # dot product of input with weights from input to hidden layer 1, add bias
        self.hidden1_input = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        # activation function used on this...
        self.hidden1_output = self.sigmoid(self.hidden1_input)
        # then take hidden layer 1 output dotted with the wieghts from the hidden layer 1 to hidden layer 2 plus bias
        self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        # activation function used on this...
        self.hidden2_output = self.sigmoid(self.hidden2_input)
        # then take sigmoid output dotted with the wieghts from the hidden layer to the output layer plus bias
        self.final_input = np.dot(self.hidden2_output, self.weights_hidden2_output) + self.bias_output
        # then use the sigmoid activation to get the output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output, learning_rate):
        """backward propagation"""
        # calculate the errors for each layer
        output_error = output - y
        hidden2_error = np.dot(output_error, self.weights_hidden2_output.T) * self.hidden2_output * (1 - self.hidden2_output)
        hidden1_error = np.dot(hidden2_error, self.weights_hidden1_hidden2.T) * self.hidden1_output * (1 - self.hidden1_output)

        # update the weights and biases from the second layer to the ouput layer
        self.weights_hidden2_output -= learning_rate * np.dot(self.hidden2_output.T, output_error)
        self.bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)

        # update the weights and bias between the two hidden layers
        self.weights_hidden1_hidden2 -= learning_rate * np.dot(self.hidden1_output.T, hidden2_error)
        self.bias_hidden2 -= learning_rate * np.sum(hidden2_error, axis=0, keepdims=True)

        # update the weights and bias for the input to the first hidden layer
        self.weights_input_hidden1 -= learning_rate * np.dot(X.T, hidden1_error)
        self.bias_hidden1 -= learning_rate * np.sum(hidden1_error, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        """Train model"""
        # set up mean square error array to track progress with each iteration
        loss_history = []
        for epoch in range(epochs):
            # forward pass
            output = self.forward(X)
            # backward pass
            self.backward(X, y, output, learning_rate)
            # calculate the loss
            # loss: binary cross-entropy error function
            loss = binary_cross_entropy(y, output)
            loss_history.append(loss)
            # periodically print the loss
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

        return loss_history

    def train_with_validation(self, X_train, y_train, X_val, y_val, epochs, learning_rate):
        """Train model"""
        # set up mean square error array to track progress with each iteration
        training_loss = []
        validation_loss = []
        for epoch in range(epochs):
            # forward pass
            output = self.forward(X_train)
            # backward pass
            self.backward(X_train, y_train, output, learning_rate)
            
            # calculate the loss
            # loss: binary cross-entropy error function
            output = self.forward(X_train)
            loss = binary_cross_entropy(y_train, output)
            training_loss.append(loss)

            # check validation set
            output_val = self.forward(X_val)
            val_loss = binary_cross_entropy(y_val, output_val)
            validation_loss.append(val_loss)
            # periodically print the loss
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Val loss: {val_loss:.4f}')

        return training_loss, validation_loss

    def predict(self, X):
        """make a prediction"""
        # first scale test data in same way train data was scaled to match weights and biases
        features = X.columns

        X_scaled = X.copy()
        X_scaled[features] = (X_scaled[features] - self.mu) / self.sigma
        output = self.forward(X_scaled.to_numpy())

        return (output > 0.5).astype(int)

    def save_weights(self, file_name='trained_weights.pkl'):
        """Save everything to a pickle file"""

        model_data = {
            'hidden_size' : self.hidden_size,
            'weights_input_hidden1' : self.weights_input_hidden1,
            'weights_hidden1_hidden2' : self.weights_hidden1_hidden2,
            'weights_hidden2_output' : self.weights_hidden2_output,
            'bias_hidden1' : self.bias_hidden1,
            'bias_hidden2' : self.bias_hidden2,
            'bias_output' : self.bias_output,
            'mu' : self.mu,
            'sigma' : self.sigma,
        }
        with open(file_name, "wb") as f:
            pickle.dump(model_data, f)
        print("Model weights saved to", file_name)

    def load_weights(self, file_name='trained_weights.pkl'):
        """Load pickle file"""
        with open(file_name, "rb") as f:
            try:
                model_data = pickle.load(f)
            except (
                pickle.UnpicklingError,
                EOFError,
                AttributeError,
                ImportError,
                IndexError
            ) as e:
                print(f"Error parsing pickle file: {e}.")
                sys.exit(1)

        hidden_size = model_data['hidden_size']
        if hidden_size != self.hidden_size:
            print("Override of hidden layer size to:", hidden_size)
        self.hidden_size = hidden_size
        self.weights_input_hidden1 = model_data['weights_input_hidden1']
        self.weights_hidden1_hidden2 = model_data['weights_hidden1_hidden2']
        self.weights_hidden2_output = model_data['weights_hidden2_output']
        self.bias_hidden1 = model_data['bias_hidden1']
        self.bias_hidden2 = model_data['bias_hidden2']
        self.bias_output = model_data['bias_output']
        self.mu = model_data['mu']
        self.sigma = model_data['sigma']

        print("Model weights loaded from", file_name)
