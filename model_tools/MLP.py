import numpy as np


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
        # then take hidden layer 1 output dotted with the wieghts from the hidden layer 1 to hidden layer 2 plus bias
        self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        # activation function used on this...
        self.hidden2_output = self.sigmoid(self.hidden2_input)
        # then take sigmoid output dotted with the wieghts from the hidden layer to the output layer plus bias
        self.final_input = np.dot(self.hidden2_output, self.weights_hidden2_output) + self.bias_output
        # then use the softmax activation to get the output
        self.final_output = self.softmax(self.final_input)
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

