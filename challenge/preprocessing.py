from datetime import datetime

import pandas as pd
import numpy as np

from challenge.constants import (
    DATETIME_FORMAT,
    THRESHOLD_MINUTES,
    OHE_VALUES,
    FEATURES_COLS,
)


def create_ohe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create one hot encoding features

    Args:
        data: raw data.

    Returns:
        features (pd.DataFrame): the one hot encoding features
    """
    features = pd.DataFrame(columns=FEATURES_COLS)
    for col in OHE_VALUES:
        for v in OHE_VALUES[col]:
            features[f"{col}_{v}"] = (data[col] == v).astype(int)

    return features[FEATURES_COLS]


def get_min_diff(data: pd.DataFrame) -> float:
    """
    Get the difference between the In and Out date in minutes

    Args:
        data: raw data.

    Returns:
        min_diff: minutes of delay.
    """
    fecha_o = datetime.strptime(data["Fecha-O"], DATETIME_FORMAT)
    fecha_i = datetime.strptime(data["Fecha-I"], DATETIME_FORMAT)
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60

    return min_diff


def create_min_diff(data: pd.DataFrame, name_col: str = "min_diff") -> pd.DataFrame:
    """
    Create a column in the dataset to get the minutes of delay

    Args:
        data: raw data.
        name_col: the name of the new column that will be created.

    Returns:
        data (pd.DataFrame): preprocessed data.
    """
    data[name_col] = data.apply(get_min_diff, axis=1)
    return data


def create_target(data: pd.DataFrame, target_col_name: str = "target") -> pd.DataFrame:
    """
    Check if the airplane is delayed

    Args:
        data: raw data.
        target_col_name: the name of the target column.

    Returns:
        df_target (pd.DataFrame): a dataframe that contains the target column.
    """
    data = create_min_diff(data, "min_diff")

    # "delay" was renamed to "target"
    df_target = pd.DataFrame()
    df_target[target_col_name] = np.where(data["min_diff"] > THRESHOLD_MINUTES, 1, 0)

    # Remove temporal columns
    data = data.drop(columns="min_diff")

    return df_target
