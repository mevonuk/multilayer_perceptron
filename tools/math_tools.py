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


def range_of_means(df) -> bool:
    """Returns true if range of means is
    within the average standard deviation"""
    means = df['Mean']
    ave_std = mean_column(df['Std'])

    max_diff = max_item(means) - min_item(means)

    if my_abs(max_diff) < my_abs(ave_std):
        return True
    else:
        return False


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


def count_items(col) -> int:
    """Counts non-null items in pandas series"""
    count = 0
    for item in col:
        if pd.notnull(item):
            count += 1
    return count


def sum_items(col) -> int | float:
    """Sums non-null items in pandas series"""
    s = 0
    for item in col:
        if pd.notnull(item):
            s += item
    return s


def max_item(col) -> int | float:
    """Returns maximum value in pandas series"""
    col = col.dropna()
    if len(col) > 0:
        max = col.iloc[0]
        for item in col:
            if item > max:
                max = item
        return max
    return np.nan


def min_item(col) -> int | float:
    """Returns maximum value in pandas series"""
    col = col.dropna()
    if len(col) > 0:
        min = col.iloc[0]
        for item in col:
            if item < min:
                min = item
        return min
    return np.nan


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


def quartile(q, col) -> int | float:
    """Return value at portion of series where q indicate portion"""
    col = col.dropna()
    N = len(col)
    if N == 0:
        return float('nan')
    if not (0 <= q <= 1):
        raise ValueError("q must be between 0 and 1")

    # index needs to be reset when using sort for series
    x = col.sort_values().reset_index(drop=True)

    # to match the pandas series quartile, need to
    # use interpolated quartile calculation
    # rank
    r = q * (N - 1)
    # integer part
    i = int(r)
    # fractional part
    f = r - i

    return x.iloc[i] * (1 - f) + x.iloc[i + 1] * f


def first_quartile(col) -> int | float:
    """Return first quartile of series"""
    return quartile(0.25, col)


def median_item(col) -> int | float:
    """Returns median value of series"""
    return quartile(0.5, col)


def third_quartile(col) -> int | float:
    """Return third quartile of series"""
    return quartile(0.75, col)


def rank_column(values) -> pd.Series:
    """Returns ranks of items in series"""
    values = values.dropna()
    # Convert to a list to ensure indexable
    values = list(values)
    n = len(values)

    # Pair each value with its original index
    value_index_pairs = [(val, idx) for idx, val in enumerate(values)]

    # Sort by value
    value_index_pairs.sort(key=lambda x: x[0])

    ranks = [0] * n
    i = 0
    while i < n:
        tie_val = value_index_pairs[i][0]
        tie_indices = [value_index_pairs[i][1]]
        j = i + 1
        # Find all tied values
        while j < n and value_index_pairs[j][0] == tie_val:
            tie_indices.append(value_index_pairs[j][1])
            j += 1
        # Average rank (1-indexed)
        avg_rank = (i + 1 + j) / 2
        for idx in tie_indices:
            ranks[idx] = avg_rank
        i = j
    return pd.Series(ranks)


def spearman(col1, col2):
    """Return Spearman rank correlation coefficient"""
    # Drop rows where either column is NaN
    valid = ~(col1.isna() | col2.isna())
    col1 = col1[valid]
    col2 = col2[valid]
    # Step 1: rank columns
    rank1 = rank_column(col1)
    rank2 = rank_column(col2)
    # Step 2: differences of ranks
    d = rank1 - rank2
    # Step 3: squared differences
    d_squared = d ** 2
    # Step 4: number of observations (after dropping NaNs)
    n = len(d)
    # Step 5: Spearman formula
    rho = 1 - (6 * sum_items(d_squared)) / (n * (n**2 - 1))
    return rho


def pearson(col1, col2):
    """Returns the Pearson's coeff of correlation between two series
    range (-1 to 1) value greater than abs(0.7) strong correlation"""
    # Step 1 Calculate means
    mean1 = mean_column(col1)
    mean2 = mean_column(col2)
    # Step 2: Calculate STDs
    std1 = std_column(col1)
    std2 = std_column(col2)
    # Step 3: Calculate deviations from the means
    dev1 = col1 - mean1
    dev2 = col2 - mean2
    # Step 4: Apply formula
    r_tot = sum_items(dev1*dev2) / (count_items(dev1*dev2) * std1 * std2)
    return r_tot


def skew_column(col):
    """Skew = (Mean - Median) / Standard Deviation"""
    col = col.dropna()
    mean = mean_column(col)
    std = std_column(col)
    median = median_item(col)
    skew = (mean - median) / std
    return skew


def kurtosis_column(col):
    """Kurtosis = mean of ((x - mean) / std)^4"""
    col = col.dropna()
    n = len(col)
    if n == 0:
        return float('nan')  # avoid division by zero

    mean = mean_column(col)
    std = std_column(col)
    if std == 0:
        return float('nan')  # avoid division by zero

    # Compute kurtosis
    fourth_moments = [((x - mean) / std) ** 4 for x in col]
    return sum_items(fourth_moments) / len(fourth_moments)


def excess_kurtosis(col):
    """Excess kurtosis (with respect to normal)"""
    kurt = kurtosis_column(col)
    return kurt - 3.0
