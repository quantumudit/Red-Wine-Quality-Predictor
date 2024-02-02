"""
This module contains the DataIngestion class which is responsible for fetching
and saving datasets from the UCI repository.
It reads configuration files, fetches the specified UCI dataset,
and saves the raw dataset. It also handles exceptions and logs the process.
"""

from os.path import dirname, exists, normpath

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.constants import CONFIGS
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import create_directories, read_yaml


class DataIngestion:
    """
    This class is responsible for data ingestion from the UCI repository.
    It reads the configuration files, fetches the specified UCI dataset,
    and saves the raw dataset.
    """

    def __init__(self):
        """
        Initializes the DataIngestion class. Reads the configuration files.
        """
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).data_ingestion

        # Define configuration parameters
        self.uci_data_id = self.configs.uci_dataset_id
        self.download = self.configs.download

        self.output_filepath = normpath(self.configs.external_path)

    @staticmethod
    def fetch_uci_dataset(uci_id: int) -> pd.DataFrame:
        """
        Fetches the UCI dataset with the given ID.

        Args:
            uci_id (int): The ID of the UCI dataset to fetch.

        Returns:
            pd.DataFrame: The original dataframe of the fetched dataset.
        """
        # Retrieve the dataset
        data_details = fetch_ucirepo(id=uci_id)
        original_df = data_details.data.original
        return original_df

    def save_raw_dataset(self) -> None:
        """
        Saves the raw dataset. If the dataset already exists or,
        re-download is not required, it skips the download.

        Raises:
            CustomException: If there is an error in the process,
            it raises a custom exception.
        """
        try:
            # Create directory if not exist
            create_directories([dirname(self.output_filepath)])

            # Download and save data if required
            if not exists(self.output_filepath) or self.download:
                wine_data = self.fetch_uci_dataset(uci_id=self.uci_data_id)
                wine_data.to_csv(
                    self.output_filepath, index=False, header=True, encoding="utf-8"
                )
                logger.info("Data saved at: %s", self.output_filepath)
            else:
                logger.info(
                    "The %s already exists. Skipping download", self.output_filepath
                )
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e
