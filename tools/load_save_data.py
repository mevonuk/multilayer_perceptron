# load a csv file without header into a pandas dataframe
import pandas as pd
import os
import pickle


def load(path: str) -> pd.core.frame.DataFrame:
    """takes file path of dataset to load,
    displays a message specifying the dimensions of the dataset,
    returns the dataset loaded as a pandas.DataFrame"""
    if not isinstance(path, str):
        print("Filename should be a string")
        return None
    if not os.path.isfile(path):
        print("File not found.")
        return None
    if not os.stat(path).st_size:
        print("File is empty.")
        return None

    # Check file extension
    if not path.lower().endswith('.csv'):
        print("Unsupported file format. Only CSV files are allowed.")
        return None

    try:
        # load data with pandas using most common character encoding
        data = pd.read_csv(path, encoding='utf-8', header=None)
        # print("Loading dataset of dimensions", data.shape)
        return data
    except pd.errors.ParserError:
        print("Error parsing CSV. File may not be properly formatted.")
        return None
    except UnicodeDecodeError:
        print("Encoding issue. Try a different encoding.")
        return None


def load_split_data(filename="split_datasets.pkl"):
    with open(filename, "rb") as f:
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

    return X_train, y_train, X_test, y_test


def save_split_data(X_train, y_train, X_test, y_test, filename="split_datasets.pkl"):
    # store the datasets to be stowed in pickle file
    split_datasets = {
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_test' : y_test,
    }
    with open(filename, "wb") as f:
        pickle.dump(split_datasets, f)
    print("Split datasets saved to", filename)
