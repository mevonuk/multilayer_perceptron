import sys
import numpy as np
import pickle
from .preprocessing import get_mean_std
from .math_tools import my_metrics, my_abs, binary_cross_entropy
from .plot_loss import plot_metrics


class MLP:
    """multilayer perceptron class with 2 hidden layers"""
    def __init__(self, input_size, hidden_size, output_size):
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
        # set mu and sigma to default values
        self.mu = 1.0
        self.sigma = 1.0

    def normalize_data(self, X, set_norm=False):
        """normalize the dataset"""
        features = X.columns
        if set_norm:
            # compute statistics on training set
            self.mu, self.sigma = get_mean_std(X, features)

        # apply normalization transformation to training set
        X_scaled = X.copy()
        X_scaled[features] = (X_scaled[features] - self.mu) / self.sigma
        return X_scaled

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

    def train_with_validation(
            self, X_train, y_train,
            X_val, y_val,
            epochs, learning_rate, stop_pt=5e-4):
        """Train model while tracking performance on validation set"""
        # set up arrays to track progress with each iteration
        training_loss = []
        validation_loss = []
        acc = []
        acc_val = []
        training_F1 = []
        training_precision = []
        prev_loss = None
        for epoch in range(epochs):

            # train the model:
            # forward pass
            output = self.forward(X_train)
            # backward pass
            self.backward(X_train, y_train, output, learning_rate)
            
            # calculate the loss
            # loss: binary cross-entropy error function
            output = self.forward(X_train)
            loss = binary_cross_entropy(y_train, output)
            training_loss.append(loss)

            # check the validation set
            output_val = self.forward(X_val)
            val_loss = binary_cross_entropy(y_val, output_val)
            validation_loss.append(val_loss)

            # track metrics
            out_train = self.make_prediction(X_train)
            out_val = self.make_prediction(X_val)

            t_acc, t_per, t_recall, t_F1 = my_metrics(out_train, y_train)
            v_acc, v_per, v_recall, v_F1 = my_metrics(out_val, y_val)

            training_precision.append(t_per)
            training_F1.append(t_F1)

            acc.append(t_acc)
            acc_val.append(v_acc)

            # periodically print the loss and accuracy
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}')
                print(f'Loss:      training: {loss:.4f}, validation: {val_loss:.4f}')
                print(f'Accuracy:  training: {t_acc:.4f}, validation: {v_acc:.4f}')
                print(f'Precision: training: {t_per:.4f}, validation: {v_per:.4f}')
                print(f'Recall:    training: {t_recall:.4f}, validation: {v_recall:.4f}')
                print(f'F1:        training: {t_F1:.4f}, validation: {v_F1:.4f}')

            # Early stopping if loss stabilizes
            if prev_loss is not None and my_abs(prev_loss - loss) <= stop_pt:
                break
            prev_loss = loss

        # plot curves
        plot_metrics(acc, training_precision, training_F1, title="Training metrics")

        # return histories of the loss and accuracy
        return training_loss, validation_loss, acc, acc_val

    def make_prediction(self, X):
        """make a prediction"""
        output = self.forward(X)
        return (output > 0.5).astype(int)
    
    def predict(self, X):
        """make a prediction"""
        # first scale test data in same way train data was scaled to match weights and biases
        X_scaled = self.normalize_data(X, set_norm=False)
        # return prediction and probability based on scaled data
        return self.make_prediction(X_scaled), self.forward(X_scaled)

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
