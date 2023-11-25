import pandas as pd

from typing import Tuple, Union, List
from .preprocessing import (
    create_ohe,
    create_target,
)


class DelayModel:
    def __init__(self):
        self._model = None  # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None,
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        df_features = create_ohe(data)

        if target_column:
            df_target = create_target(data, target_column)
            return (df_features, df_target)
        else:
            return df_features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        return
