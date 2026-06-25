import random as random


def split_data(X, y, test_size=0.2, random_seed=None, v=0):
    """Splitting of the dataset"""
    # step 1: shuffle indices of dataset
    # set seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)

    # check validity of test_size
    if test_size > 0.9 or test_size < 0.1:
        if v: print("Split out of range, defaulting to 0.8 / 0.2 for the train / test ratio")
        test_size = 0.2
    else:
        if v: print("spliting data with train / test ratio:", 1 - test_size, "/", test_size)

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
