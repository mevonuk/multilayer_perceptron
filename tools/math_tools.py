# mathematical tools
import pandas as pd
import numpy as np


def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """calculate the binary cross entropy"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def my_metrics(y_pred, y_true):
    """Calculate various metrics for the predicted output"""

    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()

    precision = 0
    recall = 0
    F1 = 0

    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # accuracy = np.mean(y_pred == y_true)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    if TP:
        # precision = TP / (TP + FP) if FP costly
        precision = TP / (TP + FP)

        # recall = TP / (TP + FN) if FN costly
        recall = TP / (TP + FN)

        # F1 = 2 * precision * recall / (precision + recall) both important
        F1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, F1


def my_abs(x: int | float) -> int | float:
    """Returns absolute value of x"""
    if not isinstance(x, (int, float)):
        raise TypeError("for abs, x must be int/float")
    if x >= 0:
        return x
    else:
        return -x


def sqrt(x: int | float) -> float:
    """Returns sqrt of x"""
    if not isinstance(x, (int, float)):
        raise TypeError("for sqrt, x must be int/float")
    if x < 0:
        raise ValueError("sqrt undefined for negative real numbers")
    if x == 0:
        return 0.0
    last_guess = x / 2.0
    epsilon = .00000000000001
    while True:
        guess = (last_guess + x / last_guess) / 2
        if my_abs(guess - last_guess) < epsilon:
            return guess
        last_guess = guess


def sum_items(col) -> int | float:
    """Sums non-null items in pandas series"""
    s = 0
    for item in col:
        if pd.notnull(item):
            s += item
    return s


def mean_column(col) -> int | float:
    """Compute mean of series ignoring NaNs"""
    col = col.dropna()
    N = len(col)
    if N == 0:
        return float('nan')
    return sum_items(col) / N


def std_column(col) -> int | float:
    """Compute standard deviation of series ignoring NaNs"""
    col = col.dropna()
    m = mean_column(col)
    N = len(col)
    if N == 0:
        return float('nan')
    return sqrt(sum_items((x - m) ** 2 for x in col) / N)
