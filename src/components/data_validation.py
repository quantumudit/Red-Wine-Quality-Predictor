"""
This module contains the DataValidation class which is used for validating
data according to a predefined schema and configurations. It reads the schema and
configurations from YAML files, normalizes the raw file path, and provides a method
to validate columns in the data.
"""

from os.path import normpath

import pandas as pd

from src.constants import CONFIGS, SCHEMA
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import read_yaml


class DataValidation:
    """
    A class used to validate data according to a predefined schema and configurations.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the DataValidation object.
        """
        self.schema = read_yaml(SCHEMA)
        self.configs = read_yaml(CONFIGS).data_validation

        self.external_filepath = normpath(self.configs.external_path)

    def validate_columns(self):
        """
        This method validates the columns of a dataset against a predefined schema.

        Raises:
            CustomException: Raised when the columns in the dataset do not match the
            required columns from the schema.
        """
        # Required columns in the dataset
        required_cols = self.schema.external_data_schema.keys()

        # Available columns in the raw dataset
        external_data = pd.read_csv(self.external_filepath)
        external_data_cols = external_data.columns.tolist()

        # Check if the columns are equal or, not
        if sorted(required_cols) == sorted(external_data_cols):
            logger.info("Total columns in the dataset: %s", len(required_cols))
            logger.info("Dataset columns:\n%s", required_cols)
            logger.info("Data Column Validation Successful")
        else:
            logger.warning("Data Column Validation Unsuccessful")
            if len(required_cols) == len(external_data_cols):
                logger.warning("Number of columns is same. Check column names")
                logger.warning("Required column names:\n%s", required_cols)
                logger.warning("Dataset column names:\n%s", external_data_cols)
            else:
                logger.warning("Number of column mismatch occurred")
                logger.warning("Required columns: %s", len(required_cols))
                logger.warning("Dataset columns: %s", len(external_data_cols))

            logger.error("Data Column Validation Successful")
            raise CustomException("Data Validation Error. Check Logs")
