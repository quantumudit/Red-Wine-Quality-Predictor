"""
wip

- take the input features
- preprocess it with preprocessor
- predict with the model
- output score
"""

from os.path import normpath

import pandas as pd

from src.constants import CONFIGS
from src.utils.basic_utils import load_joblib, read_yaml


class ModelPrediction:
    """_summary_"""

    def __init__(self):
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).model_prediction

        # Input file path
        self.preprocessor_path = normpath(self.configs.preprocessor_path)
        self.model_path = normpath(self.configs.model_path)

    def predict(self, data: pd.DataFrame) -> float:
        """_summary_

        Args:
            data (pd.DataFrame): _description_

        Returns:
            float: _description_
        """
        preprocessor = load_joblib(self.preprocessor_path)
        en_model = load_joblib(self.model_path)

        normalized_data_array = preprocessor.transform(data)
        predicted_value = en_model.predict(normalized_data_array)
        return predicted_value
