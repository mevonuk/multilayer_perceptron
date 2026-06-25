import argparse
from preprocess import preprocess
from train import training
from predict import predicting


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
        type=int,
        default=0,
        choices=[1, 0],
        help="Verbose mode"
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="gd",
        choices=["gd", "nest", "adam", "compare"],
        help="Type of optimizer"
    )

    args = parser.parse_args()

    program_mode = args.program_mode
    hidden_size = args.hidden_size
    max_epochs = args.max_epochs
    learn_rate = args.learn_rate
    split_size = args.split_size
    verbose = args.verbose
    optimizer = args.optimizer

    print("Running program in program mode:", program_mode)
    
    if program_mode == "pre_process":
        preprocess(split_size)

    elif program_mode == "train":
        training(max_epochs, learn_rate, optimizer, hidden_size, verbose)

    elif program_mode == "predict":
        predicting(optimizer, verbose)

    else:
        preprocess(split_size)
        training(max_epochs, learn_rate, optimizer, hidden_size, verbose)
        predicting(optimizer, verbose)


if __name__ == "__main__":
    main()
