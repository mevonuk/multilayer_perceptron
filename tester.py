import argparse
from preprocess import preprocess
from train import training
from predict import predicting


def main():
    """Read in preferences and direct program as appropriate"""
    parser = argparse.ArgumentParser(description="process, train, predict MLP model")

    parser.add_argument(
        "--activation",
        type=str,
        default="softmax",
        choices=["softmax", "sigmoid"],
        help="Output activation function"
    )

    parser.add_argument(
        "--pdf",
        type=int,
        default=0,
        choices=[0, 1],
        help="Save figures as PDFs"
    )

    parser.add_argument(
        "--program_mode",
        type=str,
        default="all",
        choices=["preprocess", "train", "predict", "all"],
        help="Mode of program execution"
    )

    parser.add_argument(
        "--hidden_layers",
        type=str,
        default="8,8,8",
        help="Comma-separated list of neurons in each hidden layer"
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
        default=100000,
        help="Maximum number of Epochs"
    )

    parser.add_argument(
        "--learn_factor",
        type=float,
        default=1.0,
        help="Learning rate factor: multiple of default learning rate for an optimizer"
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
        choices=["gd", "nest", "adam", "rms", "compare"],
        help="Type of optimizer"
    )

    args = parser.parse_args()

    program_mode = args.program_mode
    hidden_layers = [int(x) for x in args.hidden_layers.split(",")]
    max_epochs = args.max_epochs
    learn_factor = args.learn_factor
    split_size = args.split_size
    verbose = args.verbose
    optimizer = args.optimizer
    activation = args.activation
    save_figs = args.pdf

    print("Running program in program mode:", program_mode)
    
    if program_mode == "preprocess":
        preprocess(split_size, activation, verbose)

    elif program_mode == "train":
        training(
            max_epochs,
            learn_factor,
            optimizer,
            hidden_layers,
            activation,
            save_figs,
            verbose
        )

    elif program_mode == "predict":
        predicting(optimizer, activation, verbose)

    else:
        preprocess(split_size, activation, verbose)
        training(
            max_epochs,
            learn_factor,
            optimizer,
            hidden_layers,
            activation,
            save_figs,
            verbose)
        predicting(optimizer, activation, verbose)


if __name__ == "__main__":
    main()
