import numpy as np
import pickle


def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class MLP:
    """multilayer perceptron class with 2 hidden layers"""
    def __init__(self, input_size, hidden_size, output_size):
        """randomly initializes weights between 4 layers:
        input to hidden1,
        hidden1 to hidden2,
        hidden2 to output
        and sets biases to zero"""
        self.weights_input_hidden1 = np.random.randn(input_size, hidden_size)
        self.weights_hidden1_hidden2 = np.random.randn(hidden_size, hidden_size)
        self.weights_hidden2_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden1 = np.zeros((1, hidden_size))
        self.bias_hidden2 = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        """sigmoid function"""
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        """softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        """forward propagation"""
        # X is the input
        # dot product of input with weights from input to hidden layer 1, add bias
        self.hidden1_input = np.dot(X, self.weights_input_hidden1) + self.bias_hidden1
        # activation function used on this...
        self.hidden1_output = self.sigmoid(self.hidden1_input)
        # print(self.hidden1_output)
        # then take hidden layer 1 output dotted with the wieghts from the hidden layer 1 to hidden layer 2 plus bias
        self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        # print(self.hidden2_input)
        # activation function used on this...
        self.hidden2_output = self.sigmoid(self.hidden2_input)
        # print(self.hidden2_output)
        # then take sigmoid output dotted with the wieghts from the hidden layer to the output layer plus bias
        self.final_input = np.dot(self.hidden2_output, self.weights_hidden2_output) + self.bias_output
        # then use the sigmoid activation to get the output
        self.final_output = self.sigmoid(self.final_input)
        # print(self.final_output)
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
        for epoch in range(epochs):
            # forward pass
            output = self.forward(X)
            # backward pass
            self.backward(X, y, output, learning_rate)
            # periodically calculate and print the loss
            if (epoch + 1) % 1000 == 0:
                # loss: binary cross-entropy error function
                print(max((output)), min((output)))
                print(max(np.log(output)), min(np.log(output)), max(np.log(1 - output)), min(np.log(1 - output)))
                loss = -np.sum(y * np.log(output) + (1 - y) * np.log(1 - output)) / X.shape[0]
                print(binary_cross_entropy(y, output))
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    def predict(self, X):
        # make a prediction
        output = self.forward(X)
        return (output > 0.5).astype(int)

    def save_weights(self, file_name='trained_weights.pkl'):
        # Save everything to pickle file
        model_data = {
            'weights_input_hidden1' : self.weights_input_hidden1,
            'weights_hidden1_hidden2' : self.weights_hidden1_hidden2,
            'weights_hidden2_output' : self.weights_hidden2_output,
            'bias_hidden1' : self.bias_hidden1,
            'bias_hidden2' : self.bias_hidden2,
            'bias_output' : self.bias_output
        }
        with open(file_name, "wb") as f:
            pickle.dump(model_data, f)
        print("Model weights saved to", file_name)

    def load_weights(self, file_name='trained_weights.pkl'):
        # Load pickle file
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

        self.weights_input_hidden1 = model_data['weights_input_hidden1']
        self.weights_hidden1_hidden2 = model_data['weights_hidden1_hidden2']
        self.weights_hidden2_output = model_data['weights_hidden2_output']
        self.bias_hidden1 = model_data['bias_hidden1']
        self.bias_hidden2 = model_data['bias_hidden2']
        self.bias_output = model_data['bias_output']
        print("Model weights loaded from", file_name)
