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


def label_data(df: pd.DataFrame) -> pd.DataFrame:
    """adds column labels to dataframe"""
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
    df.columns = labels
    return df


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
