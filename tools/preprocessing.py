import numpy as np
import pandas as pd
from .math_tools import mean_column, std_column


def hot_code(
    df: pd.core.frame.DataFrame,
    positive_class: str,
    col: str,
    new_col: str,
    ) -> pd.core.frame.DataFrame:
    """One-hot coding for two class data"""
    df[new_col] = (df[col] == positive_class).astype(int)
    return df


def label_data(df: pd.core.frame.DataFrame, labels: list) -> pd.core.frame.DataFrame:
    """adds column labels to dataframe"""
    df.columns = labels
    return df


def normalized_features(df, cols):
    """create normalized data"""
    new_names = []
    feature_means = []
    feature_stds = []

    for col in cols:
        mean_col = mean_column(df[col])
        std_col = std_column(df[col])
        df[col+'_norm'] = (df[col] - mean_col) / std_col
        new_names.append(col+'_norm')
        feature_means.append(mean_col)
        feature_stds.append(std_col)

    return df, new_names, feature_means, feature_stds


def get_mean_std(df, cols):
    """get means and stds"""
    feature_means = []
    feature_stds = []

    for col in cols:
        mean_col = mean_column(df[col])
        std_col = std_column(df[col])
        feature_means.append(mean_col)
        feature_stds.append(std_col)

    feature_stds = np.asarray(feature_stds)
    feature_means = np.asarray(feature_means)

    return feature_means, feature_stds
