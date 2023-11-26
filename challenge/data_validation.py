import pandas as pd
from typing import List

from .constants import (
    VALID_VALUES,
    OHE_VALUES,
)


def get_invalid_columns(data: pd.DataFrame()) -> List:
    """
    Get columns that contains unexpected categories

    Args:
        data: raw data.

    Returns:
        invalid_cols (list): invalid columns
    """
    invalid_cols = []
    for col in OHE_VALUES:
        invalid_values = [e for e in data[col].values if e not in VALID_VALUES[col]]
        if len(invalid_values):
            invalid_cols.append(col)

    return invalid_cols
