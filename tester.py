from tools.load_data import load
from tools.split import split_data
from tools.preprocessing import hot_code, label_data
from tools.preprocessing import normalized_features, get_mean_std
from plotting.pairplot import pairplotter
from model_tools.MLP import MLP, binary_cross_entropy
import numpy as np
import argparse
import pickle


def main():
    """Based on selected mode:
    pre_process: cleans data and splits it into training and testing sets;
    train: trains the MLP model on the training data;
    predict: predicts the oucome of the test data and provides accuracy;
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
        default=10000,
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
        type=bool,
        default=False,
        choices=[True, False],
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
    print("with", hidden_size, "neurons in the each of the two hidden layers")
    

    if program_mode in ("pre_process", "all"):
        print("Starting pre-processing of data...")
        data = None
        try:
            dataset = "data/data.csv"

            # load dataset
            data = load(dataset)
            # label columns
            labels = [
                'ID',
                'diagnosis',
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
            data = label_data(data, labels)

            # chosen features
            features = [
                'radius_mean',
                'texture_mean',
                'perimeter_mean',
                'fractal_dim_mean',
                'radius_std',  # okay
                'perimeter_std',  # good
                'concave_pts_std',  # good
                'symmetry_std',  # good
                'radius_worst',  # good
                'texture_worst',  # okay
                'concavity_worst',  # good
                'concave_pts_worst',  # good
                'symmetry_worst',  # good
            ]

            # one-hot code the diagnosis
            print("One-hot encoding diagnosis...")
            diagnoses = ['M', 'B']
            data = hot_code(data, 'M', 'diagnosis', 'one_hot')

            # extracting X and y data arrays
            # y is the diagnosis one-hot coded
            # X contains the normalized features: chosen based on the graph analysis
            X = data.loc[:, data.columns.intersection(features)]
            y = data.loc[:, data.columns.intersection(['one_hot'])]

            # Split the dataset into test and train sets
            if split_size > 0.9 or split_size < 0.1:
                print("Split out of range, defaulting to 0.8 / 0.2 for the train / test ratio")
                split_size = 0.2
            else:
                print("spliting data with train / test ratio:", 1 - split_size, "/", split_size)

            X_train, X_test, y_train, y_test = split_data(X, y, test_size=split_size, random_seed=42)

            # store the datasets to be stowed in pickle file
            split_datasets = {
                'X_train' : X_train,
                'X_test' : X_test,
                'y_train' : y_train,
                'y_test' : y_test,
            }
            with open("split_datasets.pkl", "wb") as f:
                pickle.dump(split_datasets, f)
            print("Split datasets saved to split_datasets.pkl")

        except (TypeError, Exception, KeyboardInterrupt) as e:
            print(e)
        
    if program_mode in ("train", "predict"):
        print("maximum number of epochs:", max_epochs)
        print("Learning rate:", learn_rate)
        try:
            with open("split_datasets.pkl", "rb") as f:
                try:
                    split_datasets = pickle.load(f)
                except (
                    pickle.UnpicklingError,
                    EOFError,
                    AttributeError,
                    ImportError,
                    IndexError
                ) as e:
                    print(f"Error parsing pickle file: {e}.")
                    sys.exit(1)

            X_train = split_datasets['X_train']
            X_test = split_datasets['X_test']
            y_train = split_datasets['y_train']
            y_test = split_datasets['y_test']

            print("Split data loaded")

        except (TypeError, Exception, KeyboardInterrupt) as e:
            print(e)

    if program_mode in ("train", "all"):

        try:
            print("initializing and training MLP model...")

            # feature normalization
            # compute statistics on training set
            features = X_train.columns
            mu, sigma = get_mean_std(X_train, features)

            # apply normalization transformation to training set
            X_train[features] = (X_train[features] - mu) / sigma

            # (optional) plot normalized features if necessary
            if verbose: pairplotter(data, features, 'diagnosis')

            # make the MLP specifying size of hidden and output layers
            mp_test = MLP(X_train.shape[1], hidden_size, 1, mu, sigma)

            # train the model
            mp_test.train(X_train.to_numpy(), y_train.to_numpy(), max_epochs, learn_rate)

            # save the weights and biases
            mp_test.save_weights()
            print("training weights and normalization saved.")

        except (TypeError, Exception, KeyboardInterrupt) as e:
            print(e)

    if program_mode in ("predict", "all"):
        try:
            # make the MLP specifying size of hidden and output layers
            mp_test = MLP(X_test.shape[1], hidden_size, 1)

            # load weights
            print("Loading saved weights...")
            mp_test.load_weights()

            # make a prediction using the trained weights and the test data
            print("Making prediction...")
            output_prediction = mp_test.predict(X_test)

            output_prob = mp_test.forward(X_test.to_numpy())
            print(
                "Binary cross entropy of the prediction:",
                binary_cross_entropy(y_test.to_numpy(), output_prob))

            # compare prediction to real values
            accuracy = np.mean(output_prediction == y_test.to_numpy())
            print("Accuracy of the prediction:", accuracy)
    
        except (TypeError, Exception, KeyboardInterrupt) as e:
            print(e)


if __name__ == "__main__":
    main()
