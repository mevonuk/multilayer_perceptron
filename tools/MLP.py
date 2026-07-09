import sys
import numpy as np
import pickle
from .preprocessing import get_mean_std
from .math_tools import my_metrics, binary_cross_entropy
from .math_tools import sigmoid, softmax, relu


class MLP:
    """multilayer perceptron class with 2 hidden layers
    using Nesterov momentum as the optimizer"""
    def __init__(self, input_size, hidden_layers=None, optimizer='gd', activation='sigmoid'):
        """randomly initializes weights between 4 layers:
        input to hidden1,
        hidden1 to hidden2,
        hidden2 to output
        and sets biases to zero"""

        # Set output size based on activation layer
        self.activation = activation
        if activation == 'sigmoid':
            output_size = 1
        else:
            output_size = 2

        # set seed for repeatability
        np.random.seed(42)

        self.optimizer = optimizer
        if hidden_layers is None:
            hidden_layers = [10,10]

        # flexible implementation
        self.hidden_layers = hidden_layers
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        self.weights = []
        self.biases = []
        # initialize weights and biases
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(
                np.random.randn(in_size, out_size) * np.sqrt(2/(in_size+out_size))
            )
            self.biases.append(
                np.zeros((1, out_size))
            )


        # variables for Nesterov momentum
        self.v_weights = [
            np.zeros_like(W) for W in self.weights
        ]
        self.v_biases = [
            np.zeros_like(b) for b in self.biases
        ]

        self.momentum = 0.9

        # Adam parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

        # First moments
        self.m_weights = [
            np.zeros_like(W) for W in self.weights
        ]
        self.m_biases = [
            np.zeros_like(b) for b in self.biases
        ]

        # Second moments
        self.s_weights = [
            np.zeros_like(W) for W in self.weights
        ]
        self.s_biases = [
            np.zeros_like(b) for b in self.biases
        ]

        # RMSprop hyperparameters
        self.rms_beta = 0.9
        self.rms_epsilon = 1e-8

        # Running squared-gradient averages
        self.rms_weights = [
            np.zeros_like(W) for W in self.weights
        ]
        self.rms_biases = [
            np.zeros_like(b) for b in self.biases
        ]


    # --- Training and prediction ---

    def train_with_validation(
            self, X_train, y_train,
            X_val, y_val,
            epochs, learning_rate, v=1):
        """Train model while tracking performance on validation set"""
        # set up arrays to track progress with each iteration
        training_loss = []
        validation_loss = []
        acc = []
        acc_val = []
        training_F1 = []
        training_precision = []
        training_recall = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 40
        min_delta = 1e-4

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
            training_recall.append(t_recall)

            acc.append(t_acc)
            acc_val.append(v_acc)

            # periodically print the loss and accuracy
            if v and ((epoch + 1) % 10 == 0):
                print(f'Epoch {epoch+1}')
                print(f'Loss:      training: {loss:.4f}, validation: {val_loss:.4f}')
                print(f'Accuracy:  training: {t_acc:.4f}, validation: {v_acc:.4f}')
                print(f'Precision: training: {t_per:.4f}, validation: {v_per:.4f}')
                print(f'Recall:    training: {t_recall:.4f}, validation: {v_recall:.4f}')
                print(f'F1:        training: {t_F1:.4f}, validation: {v_F1:.4f}')

            # Early stopping if validation loss stabilizes
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                
                patience_counter = 0

                # Save best weights
                self.save_weights()
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        # load best weights
        self.load_weights()

        print("total number of epochs run:", epoch)

        # return histories of the loss and accuracy
        return training_loss, validation_loss, acc, acc_val


    def make_prediction(self, X):
        """make a prediction"""
        output = self.forward(X)
        if self.activation == 'sigmoid':
            return (output > 0.5).astype(int)
        else:
            hold = np.argmax(output, axis=1)
            y = np.eye(2)[hold]
            return y
    

    def predict(self, X):
        """normalize data
        return prediction and probabilities"""
        # first scale test data in same way train data was scaled to match weights and biases
        X_scaled = self.normalize_data(X, set_norm=False).to_numpy()
        # return prediction and probability based on scaled data
        return self.make_prediction(X_scaled), self.forward(X_scaled)

    # --- Modeling tools ---

    def normalize_data(self, X, set_norm=False):
        """normalize the dataset"""
        features = X.columns
        if set_norm or (self.mu is None) or (self.sigma is None):
            # compute statistics on training set
            self.mu, self.sigma = get_mean_std(X, features)

        # apply normalization transformation to training set
        X_scaled = X.copy()
        X_scaled[features] = (X_scaled[features] - self.mu) / self.sigma
        return X_scaled

    def forward(self, X):
        """forward with relu/sigmoid activation"""
        self.activations = [X]
        self.z_values = []

        A = X

        for W, b in zip(self.weights[:-1], self.biases[:-1]):

            Z = A @ W + b
            A = relu(Z)

            self.z_values.append(Z)
            self.activations.append(A)

        # output layer
        Z = A @ self.weights[-1] + self.biases[-1]

        if self.activation == 'sigmoid':
            output = sigmoid(Z)
        else:
            output = softmax(Z)

        self.z_values.append(Z)
        self.activations.append(output)
        return output

    def backward(self, X, y, output, learning_rate):
        """backward propagation"""
        # calculate the errors for each layer
        m = X.shape[0]

        deltas = [None] * len(self.weights)

        # output delta
        deltas[-1] = output - y

        for i in reversed(range(len(self.weights)-1)):

            relu_grad = (self.z_values[i] > 0).astype(float)

            deltas[i] = (
                deltas[i+1] @ self.weights[i+1].T
            ) * relu_grad

        # compute gradients
        grad_weights = []
        grad_biases = []

        for i in range(len(self.weights)):

            grad_W = self.activations[i].T @ deltas[i] / m
            grad_b = np.mean(deltas[i], axis=0, keepdims=True)

            grad_weights.append(grad_W)
            grad_biases.append(grad_b)

        # backpropagate according to optimizer type
        if self.optimizer == 'gd':
            self.gd_grad(learning_rate, grad_weights, grad_biases)

        elif self.optimizer == 'nest':
            self.nesterov_grad(learning_rate, grad_weights, grad_biases)

        elif self.optimizer == 'adam':
            self.adam_grad(learning_rate, grad_weights, grad_biases)

        elif self.optimizer == 'rms':
            self.rmsprop_grad(learning_rate, grad_weights, grad_biases)


    # --- Load and save operations ---

    def save_weights(self, file_name='trained_weights_'):
        """Save weights, biases, norm factors to a pickle file"""

        model_data = {
            'activation' : self.activation,
            'hidden_layers' : self.hidden_layers,
            'weights' : self.weights,
            'biases' : self.biases,
            'mu' : self.mu,
            'sigma' : self.sigma,
        }
        with open(file_name + self.optimizer + '.pkl', "wb") as f:
            pickle.dump(model_data, f)


    def load_weights(self, file_name='trained_weights_'):
        """Load pickle file"""
        with open(file_name + self.optimizer + '.pkl', "rb") as f:
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

        self.activation = model_data['activation']
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.hidden_layers = model_data['hidden_layers']
        self.mu = model_data['mu']
        self.sigma = model_data['sigma']

    # --- Optimizers ---

    # --- Standard gradient descent ---

    def gd_grad(self, learning_rate, grad_weights, grad_biases):
        """does the backward propogation using the standard gradient descent method"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_weights[i]
            self.biases[i] -= learning_rate * grad_biases[i]


    # --- Nesterov momentum ---

    def nesterov_update(self, param, grad, velocity, lr):
        """Helper for the Nesterov momentum method"""
        v_prev = velocity.copy()

        velocity[:] = self.momentum * velocity - lr * grad

        param += (
            -self.momentum * v_prev
            + (1 + self.momentum) * velocity
        )

    def nesterov_grad(self, learning_rate, grad_weights, grad_biases):
        """does the backward propogation using the Nesterov momentum method"""
        # update weights and biases
        for i in range(len(self.weights)):
            self.nesterov_update(
                self.weights[i],
                grad_weights[i],
                self.v_weights[i],
                learning_rate
            )
            self.nesterov_update(
                self.biases[i],
                grad_biases[i],
                self.v_biases[i],
                learning_rate
            )


    # --- Adam optimizer ---

    def adam_update(self, param, grad, m, v, lr):
        """Adam optimizer update"""

        # update first moment (mean) estimate
        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        # update second moment (varience) estimate
        v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

        # bias corrections
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)

        # final parameter update
        param -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


    def adam_grad(self, learning_rate, grad_weights, grad_biases):
        """does the backward propogation using the Adam optimizer"""

        self.t += 1

        # updates
        for i in range(len(self.weights)):
            self.adam_update(
                self.weights[i],
                grad_weights[i],
                self.m_weights[i],
                self.s_weights[i],
                learning_rate
            )
            self.adam_update(
                self.biases[i],
                grad_biases[i],
                self.m_biases[i],
                self.s_biases[i],
                learning_rate
            )


    # --- RMSprop ---

    def rmsprop_update(self, param, grad, cache, lr):
        """RMSprop parameter update"""

        # update the running average
        cache[:] = (
            self.rms_beta * cache
            + (1 - self.rms_beta) * grad**2
        )

        # scale the learning rate
        param -= (
            lr * grad
            / (np.sqrt(cache) + self.rms_epsilon)
        )

    def rmsprop_grad(self, learning_rate, grad_weights, grad_biases):
        """does the backward propogation using RMSprop"""
        # update weights and biases
        for i in range(len(self.weights)):
            self.rmsprop_update(
                self.weights[i],
                grad_weights[i],
                self.rms_weights[i],
                learning_rate
            )
            self.rmsprop_update(
                self.biases[i],
                grad_biases[i],
                self.rms_biases[i],
                learning_rate
            )
