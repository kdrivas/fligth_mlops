import pandas as pd
import pickle

import xgboost as xgb

from typing import Tuple, Union, List
from challenge.preprocessing import (
    create_ohe,
    create_target,
)


class DelayModel:
    def __init__(self):
        self._model = None  # Model should be saved in this attribute.

    def init_model(self, model_path: str) -> None:
        """
        Load the model

        Args:
            model_path: the path where the model was saved.
        """
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

    def save_model(self, model_path: str) -> None:
        """
        Save the trained model

        Args:
            model_path: the path where the model will b saved.
        """
        if self._model is not None:
            with open(model_path, "wb") as f:
                pickle.dump(self._model, f)
        else:
            raise Exception("Model was not trained.")

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
        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=4.4402380952380955
        )
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            return [0] * len(features)
        else:
            return self._model.predict(features)
