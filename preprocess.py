from tools.load_save_data import load, save_split_data
from tools.split import split_data
from tools.preprocessing import hot_code, label_data
import numpy as np
import pandas as pd


def preprocess(split_size, activation, verbose):
    """Pre-process data:
    read in files split into training and test sets
    one-hot
    save to pickle file"""
    print("\nPre-processing data from already split grading .csv files...")
    print("Note that these files alread contain the column headers.")
    data = None
    try:
        train_dataset = "data_training.csv"
        test_dataset = "data_validation.csv"

        # load dataset
        traindata = load(train_dataset, header='infer')

        # load dataset
        testdata = load(test_dataset, header='infer')

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

        features = [
            'radius_mean',
            'radius_std',
            'radius_worst',
            'texture_mean',
            'texture_std',
            'texture_worst',
            'perimeter_mean',
            'perimeter_std',
            'perimeter_worst',
            'area_mean',
            'area_std',
            'area_worst',
            'smoothness_mean',
            'smoothness_std',
            'smoothness_worst',
            'compactness_mean',
            'compactness_std',
            'compactness_worst',
            'concavity_mean',
            'concavity_std',
            'concavity_worst',
            'concave_pts_mean',
            'concave_pts_std',
            'concave_pts_worst',
            'symmetry_mean',
            'symmetry_std',
            'symmetry_worst',
            'fractal_dim_mean',
            'fractal_dim_std',
            'fractal_dim_worst',
        ]

        # one-hot code the diagnoses in the two datasets
        if verbose: print("One-hot encoding diagnosis...")
        traindata = hot_code(traindata, activation, 'M', 'diagnosis', 'one_hot')
        testdata = hot_code(testdata, activation, 'M', 'diagnosis', 'one_hot')

        # extract X and y data arrays
        # y is the diagnosis one-hot coded
        # X contains the features chosen based on the graph analysis
        if verbose: print("Extract X and y data arrays")
        X_train = traindata.loc[:, traindata.columns.intersection(features)]
        X_test = testdata.loc[:, testdata.columns.intersection(features)]

        # extract one hot training data
        labels = traindata["label"].to_numpy()
        if activation == "sigmoid":
            y_train = labels.reshape(-1, 1)
        else:
            y_train = np.eye(2)[labels]
        y_train = pd.DataFrame(y_train)

        # extract one-hot test data
        labels = testdata["label"].to_numpy()
        if activation == "sigmoid":
            y_test = labels.reshape(-1, 1)
        else:
            y_test = np.eye(2)[labels]
        y_test = pd.DataFrame(y_test)

        print(X_test, y_test)

        # store the datasets to be stowed in a pickle file
        save_split_data(X_train, y_train, X_test, y_test, activation)

    except (TypeError, Exception, KeyboardInterrupt) as e:
        print(e)
