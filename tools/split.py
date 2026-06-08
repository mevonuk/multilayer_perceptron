import random as random


def split_data(X, y, test_size=0.2, random_seed=None):
    """Splitting of the dataset"""
    # step 1: shuffle indices of dataset
    # set seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)

    # create list of indices
    n_samples = len(X)
    indices = list(range(n_samples))

    # shuffle indices
    random.shuffle(indices)

    # split the dataset indicies based on a ratio
    split_index = int(n_samples * (1 - test_size))

    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    # apply split to dataset
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test
