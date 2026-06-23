from tools.load_save_data import load, load_split_data, save_split_data
from tools.split import split_data
from tools.preprocessing import hot_code, label_data
from tools.preprocessing import get_mean_std
from plotting.pairplot import pairplotter
from tools.plot_loss import plot2
from tools.math_tools import my_metrics, binary_cross_entropy
from tools.MLP import MLP
import argparse
import pickle
import sys
import pandas as pd


def main():
    """Based on selected mode:
    pre_process: cleans data and splits it into training and testing sets;
    train: trains the MLP model on the training data;
    predict: predicts the oucome of the test data and provides the accuracy;
    all: does all three in one go"""
    parser = argparse.ArgumentParser(description="process, train, predict MLP model")

    parser.add_argument(
        "--program_mode",
        type=str,
        default="all",
        choices=["pre_process", "train", "predict", "all"],
        help="Mode of program execution"
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=10,
        help="Number of neurons in the hidden layers"
    )

    parser.add_argument(
        "--split_size",
        type=float,
        default=0.2,
        help="Amount of train to test split (between 0.1 and 0.9)"
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of Epochs"
    )

    parser.add_argument(
        "--learn_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        choices=[1, 0],
        help="Verbose mode"
    )

    args = parser.parse_args()

    program_mode = args.program_mode
    hidden_size = args.hidden_size
    max_epochs = args.max_epochs
    learn_rate = args.learn_rate
    split_size = args.split_size
    verbose = args.verbose

    print("Running program in program mode:", program_mode)
    if program_mode in ["train", "predict", "all"]:
        print("with", hidden_size, "neurons in the each of the two hidden layers")
    

    if program_mode in ("pre_process", "all"):
        print("\nStarting pre-processing of data...")
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
            print("One-hot encoding diagnosis...")
            data = hot_code(data, 'M', 'diagnosis', 'one_hot')

            # extract X and y data arrays
            # y is the diagnosis one-hot coded
            # X contains the normalized features: chosen based on the graph analysis
            print("Extract X and y data arrays")
            X = data.loc[:, data.columns.intersection(features)]
            y = data.loc[:, data.columns.intersection(['one_hot'])]

            # Split the dataset into test and train sets
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=split_size, random_seed=42)

            # store the datasets to be stowed in a pickle file
            save_split_data(X_train, y_train, X_test, y_test)

        except (TypeError, Exception, KeyboardInterrupt) as e:
            print(e)

    if program_mode in ("train", "predict"):
        print("\nLoading the split datasets...")
        X_train, y_train, X_test, y_test = load_split_data()
        print("Split data loaded")


    if program_mode in ("train", "all"):
        print("\nStarting training of model...")
        print("maximum number of epochs:", max_epochs)
        print("Learning rate:", learn_rate)

        try:
            print("initializing and training MLP model...")

            # make the MLP specifying size of hidden and output layers
            mp_test = MLP(X_train.shape[1], hidden_size, 1)

            # feature normalization
            print("Normalizing training set...")
            X_train_norm = mp_test.normalize_data(X_train, set_norm=True)

            # (optional) plot normalized features if necessary
            if verbose:
                print("Plotting normalized data...")
                features = X_train_norm.columns
                features_to_plot = features.to_list()
                data_to_plot = pd.merge(
                    X_train_norm, y_train,
                    right_index=True, left_index=True)
                pairplotter(data_to_plot, features_to_plot, 'one_hot', save_fig=True)

            y_validation = y_test.copy()
            # normalize validation data using norm values of train set
            X_validation_norm = mp_test.normalize_data(X_test, set_norm=False)

            # train while tracking performance on validation set
            train_loss, val_loss, train_acc, val_acc = mp_test.train_with_validation(
                X_train_norm.to_numpy(), y_train.to_numpy(),
                X_validation_norm.to_numpy(), y_validation.to_numpy(),
                max_epochs, learn_rate)
            plot2(train_loss, val_loss, 'Loss')
            plot2(train_acc, val_acc, 'Accuracy')

            # save the weights and biases
            mp_test.save_weights()
            print("training weights and normalization saved.")

        except (TypeError, Exception, KeyboardInterrupt) as e:
            print(e)

    if program_mode in ("predict", "all"):
        print("\nPredicting using test dataset...")
        try:
            # make the MLP specifying size of hidden and output layers
            mp_test = MLP(X_test.shape[1], hidden_size, 1)

            # load weights
            print("Loading saved weights...")
            mp_test.load_weights()

            # make a prediction using the trained weights and the test data
            print("Making prediction...")
            output_prediction, output_prob = mp_test.predict(X_test)

            print(
                "Binary cross entropy of the prediction:",
                binary_cross_entropy(y_test.to_numpy(), output_prob))

            # compare prediction to real values
            print("\nMetrics of the final prediction:")
            accuracy, precision, recall, F1 = my_metrics(
                output_prediction, y_test.to_numpy())
            print("accuracy :", accuracy)
            print('precision:', precision)
            print('recall   :', recall)
            print('F1       :', F1)
    
        except (TypeError, Exception, KeyboardInterrupt) as e:
            print(e)


if __name__ == "__main__":
    main()
