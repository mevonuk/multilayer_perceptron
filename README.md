# Multilayer Perceptron

## Overview

Introduction to artificial neural networks implementing multilayer perceptron to predict a malignant or benign cancer diagnosis for cell masses.

### Contributor
- M. Evonuk (https://github.com/mevonuk)

## Tools
- Coding Language: Python

## Multilayer perceptron (MLP) Concepts

MLP is a type of feedforward neural network with fully connected neurons with non-linear activation functions, such as sigmoid, hyperbolic tangent, or rectified linear unit (ReLU). The network is organized into layers with one or more hidden layers between the input and output layers. A perceptron has multiple weighted input connections, an activation function, and a single output. The output is obtained by calculating the weighted sum of the inputs (the outputs of the previous layer) and applying the activation function on the weighted sum. The output of the activation function is the threshold above which the neuron is activated.

### Feedforward

Feedforward networks are networks in which information flows in a single direction with inputs multiplied by weights to obtain outputs.

### Backpropagation

Backpropagation involves computing the gradient of the loss with respect to the weights of a network for a given input-output example. That is, the process of calculating and adjusting the weights to reach an optimal value.

## Dataset

The dataset consists of a csv file with 32 columns. The features describe the characteristics of cell nuclei of breast masses extracted with fine-needle aspiration. The diagnosis column has values of M or B (malignant or benign, respectively); this is the column to be predicted.

Number of instances: 569

Number of attributes: 32 (ID, diagnosis, 30 real-valued input features)

Attribute information:
1) ID number
2) Diagnosis (M = malignant, B = benign)

Ten real-valued features are computed for each cell nucleus (mean, standard error, worst (mean of 3 largest)):
1) radius (mean of distances from center to points on the perimeter)
2) texture (standard deviation of gray-scale values)
3) perimeter
4) area
5) smoothness (local variation in radius lengths)
6) compactness (perimeter^2 / area - 1.0)
7) concavity (severity of concave portions of the contour)
8) concave points (number of concave portions of the contour)
9) symmetry 
10) fractal dimension ("coastline approximation" - 1)

Class distribution: 357 benign, 212 malignant

## Preprocessing

### Labeling columns

Data columns are labeled accordingly and scatterplots of the different feature combinations are made.

Pair plots of the mean feature values reveal that the benign and malignant classes are best differentiated for:
- 'radius_mean',
- 'texture_mean',
- 'perimeter_mean',
- 'fractal_dim_mean',
- 'radius_std',
- 'perimeter_std',
- 'concave_pts_std',
- 'symmetry_std',
- 'radius_worst',
- 'texture_worst',
- 'concavity_worst',
- 'concave_pts_worst',
- 'symmetry_worst',

with the other features showing more overlap between the two classes.

To view the pair plots using the command line, run:
- python plotter.py --subset worst

Choices for data subset are 'final', 'mean', 'std', and 'worst'.

### One-hot coding and feature normalization

The dataset is split into X containing the features of interest (see above) and y containing the one-hot encoding of the diagnosis column with malignant = 1.

The features in X are then normalized because gradient-based optimizations are sensitive to scale and convergence may be slow or inconsistent when using non-normalized features.

### Manual splitting of the dataset

The use of machine learning libraries is prohibited; therefore, the dataset needs to be split manually.

The two data subsets X and y are split into 80% training / 20% testing sets by default. A random split is essential, so the indices need to be shuffled prior to splitting.

## MLP class

A class is created to hold the weights and biases of the layers, as well as to provide routines to make forward and backward training passes. The class can also save the state of the weights/biases/normalization to a pickle file and load a pickle file with said weights, biases, and normalization terms.

The class is hard coded to have an input layer, two hidden layers, and an output layer.

## Program modes

The main program can be run in different program modes: pre_process; train; predict; and all.

### pre_process

- hot codes the diagnosis field
- splits the data into train and test sets
- saves the split data to a pickle file

### train

- loads the split data
- initializes and trains the MLP model
- early stopping is implemented
- plots training metrics (accuracy, precision, recall, F1) versus Epoch (compare mode)
- plots loss and accuracy versus epoch for the training and validation sets (individual optimizers)
- saves weights and biases of trained model to a pickle file

### predict

- loads the split data
- loads the saved MLP model weights and biases
- makes a prediction for the test set
- gives final metrics for the final prediction

## Optimizers

The default optimizer is standard gradient descent ("gd"). Also available are momentum-based optimizers, RMSprop ("rms") and Nesterov ("nest) momentum, as well as Adam ("adam). The  various optimizers return similar results but the standard gradient descent method takes much longer to converge and has slightly lower accuracy, precision, recall, and F1 metrics than the other methods.

### Standard gradient descent

Repeated steps are taken in the opposite direction of the gradient (the direction of steepest descent). This is a first-order iterative algorithm for minimizing the differential multivariate function.

### Momentum-based optimizers

 Momentum-based optimizers (such as RMSprop and Nesterov momentum) accelerate gradient descent using a moving average of past gradients. This reduces oscillations and speeds convergence.

Formula:
- v_(t+1) = beta * v_t + (1 - beta) * grad L(w_t)
- w_(t+1) = w_t - eta * v_(t+1)

where:
- v_t is the velocity (running average of gradients)
- beta is the momentum factor (between 0 and 1, how much past gradients are remembered)
- grad L(w_t) is the current gradient of the loss function
- eta is the learning rate (size of the step taken in each update)

#### RMSprop (root mean square propagation)

RMSprop uses an exponentially weighted moving average of the squared gradients to prevent the learning rate from decreasing too quickly.

Formula:
- v_(t+1) = beta * v_t + (1 - beta) * (grad L(w_t))^2
- w_(t+1) = w_t - eta / (sqrt(v_(t+1)) + epsilon) * grad L(w_t)

#### Nesterov momentum

Nesterov momentum adds a momentum term to the update rule of the gradient descent to mitigate problems with oscillation, where the momentum term is a weighted average of the past gradients with the weighting decreasing exponentially as gradients get further away in time.

- computes gradient at future position instead of current position

Formula:
- $v_(t+1)$ = beta * v_t + grad L(w_t - eta * beta * v_t)
- w_(t+1) = w_t - eta * v_(t+1)

### Adam optimizer

Adam (adaptive moment estimation) combines momentum and RMSprop techniques to adjust learning rates during training.

## To run

To set up virtual environment run:
- make setup

Then, activate the virtual environment.

To show a pair plot of the final features, run:
- make plot

To run the program (preprocessing, training, and prediction), enter:
- make run PROGRAM_MODE=program_mode OPTIMIZER=optimizer

In the chosen program_mode ("pre_process", "train", "predict", or "all") with the chosen optimizer ("gd", "nest", "adam", or "rms").

Other default running modes from Makefile:
- make preprocess OPTIMIZER=optimizer
- make train OPTIMIZER=optimizer
- make predict OPTIMIZER=optimizer

To compare the four optimizers, run:
- make compare

It is also possible to run the program from the command line with additional options:
- python tester.py --program_mode all --optimizer rms --hidden_size 10 --split_size 0.2 --max_epochs 10000 --learn_factor 0.1 --verbose 1

where:
- hidden_size indicates the number of neurons in the hidden layer
- split_size indicates the size of the test set with respect to the train set (between 0.1 and 0.9)
- max_epochs indicates the maximum number of epochs (may not be reached if early stopping criteria are met)
- learn_factor indicates the multiple of the learning rate for the optimizer (i.e., learning rate = learn_factor * default learning rate). Default learning rates are:
    - 0.01 for standard gradient descent and Nesterov momentum
    - 0.001 for Adam and RMSprop
- verbose indicates verbose mode (1 or 0)

When you are finished, deactivate the virtual environment and run:
- make clean

This is will remove the pickle files, virtual environment, and pycache files.
