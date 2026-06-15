# Multilayer Perceptron

## Overview

Introduction to artificial neural networks implementing a multilayer perceptron to predict a malignant or benign cancer diagnosis for cell masses.

### Contributor
- M. Evonuk (https://github.com/mevonuk)


## Tools
- Coding Language: Python

## Multilayer perceptron (MLP) Concepts

Type of feed forward nueral network with fully connected neurons with non linear activation functions, such as sigmoid, hyperbolic tangent, or rectified linear unit (ReLU). The network is organized in layers with one or more hidden layers between the input and output layers. A perceptron has multiple weighted input connections, an activation function, and a single output. The ouput is obtained by calculating the weighted sum of the inputs (the outputs of the previous layer) and applying the activation function on the weighted sum. The ouput of the activation function is the threshold above which the neuron is activated.

### Feedforward

Feedforward networks are networks in which information flows in a single direction with inputs multiplied by weights to obtain ouputs.

### Backpropagation

Backpropagation involves computing the gradient of the loss with respect to the weights of a network for a given input-output example. That is, the process of calculating and adjusting the weights to reach an optimal value.

### Gradient descent

A mathematical optimization method. Repeated steps are taken in the opposite direction of the gradient (the direction of steepest descent). This is a first-order iterative algorithm for minimizing a differential multivariate function.

## Dataset

The dataset conssits of a csv file with 32 columns. The features describe the characteristics of cell nuclei of breast masses extracted with fine-needle aspiration. The diagnosis column has values of M or B (malignant or benign, respectively); this is the column to be predicted.

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

Scatterplots and pairplots of the mean feature values reveal that the benign and malignant classes are best differentiated for:
- 'radius_mean',
- 'texture_mean',
- 'perimeter_mean',
- 'fractal_dim_mean',
- 'radius_std',  # okay
- 'perimeter_std',  # good
- 'concave_pts_std',  # good
- 'symmetry_std',  # good
- 'radius_worst',  # good
- 'texture_worst',  # okay
- 'concavity_worst',  # good
- 'concave_pts_worst',  # good
- 'symmetry_worst',  # good
with the other features showing some overlap.

### One-hot coding and feature normalization

The dataset is split into X containing the features of interest (see above) and y containing the one-hot encoding of the diagnosis column with malignant = 1.

The features in X are then normalized because gradient-based optimizations are sensitive to scale and convergence may be slow or inconsistant.

### Manual splitting of the dataset

The use of machine learning libraries is prohibited; therefore, the dataset needs to be split manually.

The two data sub sets X and y are split into 80% training / 20% testing sets. A random split is essential, so the indices need to be shuffled prior to spliting.

## MLP class

A class is created to hold the weights and biases of the layers, as well as to provide routines to make forward and backward passes. The class can also save the state of the weights/biases to a pickle file, and load a pickle file with said weights and biases.

Next:
- graphing during training for loss and accuracy
- early stopping
- A more complex optimization function (for example: Nesterov momentum,
RMSprop, Adam, ...).
- Evaluate the learning phase with multiple metrics.
- A history of the metrics obtained during training.
- A display of multiple learning curves on the same graph (really useful to compare
different models).