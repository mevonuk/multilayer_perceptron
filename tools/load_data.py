# load a csv file without header into a pandas dataframe
import pandas as pd
import os


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
