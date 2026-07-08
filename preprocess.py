from tools.load_save_data import load, save_split_data
from tools.split import split_data
from tools.preprocessing import hot_code, label_data
import numpy as np
import pandas as pd


def preprocess(split_size, activation, verbose):
    """Pre-process data:
    split into training and test sets
    save to file"""
    print("\nPre-processing data...")
    data = None
    try:
        dataset = "data/data.csv"

        # load dataset
        data = load(dataset)
        # label columns
        data = label_data(data)

        # chosen features
        features = [
            'radius_mean',
            'texture_mean',
            'perimeter_mean',
            'fractal_dim_mean',
            'perimeter_std',
            'concave_pts_std',
            'radius_worst',
            'concavity_worst',
            'concave_pts_worst',
            'symmetry_worst',
        ]

        # one-hot code the diagnosis
        if verbose: print("One-hot encoding diagnosis...")
        data = hot_code(data, activation, 'M', 'diagnosis', 'one_hot')

        # extract X and y data arrays
        # y is the diagnosis one-hot coded
        # X contains the normalized features: chosen based on the graph analysis
        if verbose: print("Extract X and y data arrays")
        X = data.loc[:, data.columns.intersection(features)]

        labels = data["label"].to_numpy()

        if activation == "sigmoid":
            y = labels.reshape(-1, 1)
        else:
            y = np.eye(2)[labels]
            
        # Split the dataset into test and train sets
        X_train, X_test, y_train, y_test = split_data(
            X, pd.DataFrame(y), test_size=split_size, random_seed=42, v=verbose)

        # store the datasets to be stowed in a pickle file
        save_split_data(X_train, y_train, X_test, y_test, activation)

    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)
