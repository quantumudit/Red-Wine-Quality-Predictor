"""
This module contains a class for training a machine learning model using
ElasticNet regression. The class reads in configuration files, prepares the model
with specified hyperparameters, trains the model on a given dataset, and
saves the trained model.
"""

from os.path import dirname, normpath

import numpy as np
from sklearn.linear_model import ElasticNet

from src.constants import CONFIGS, PARAMS
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import create_directories, read_yaml, save_as_joblib


class ModelTrainer:
    """
    A class used to train a machine learning model using ElasticNet regression.

    ...

    Attributes
    ----------
    configs : dict
        A dictionary containing the configurations for the model trainer.
    params : dict
        A dictionary containing the parameters for the ElasticNet model.
    random_seed : int
        The seed for the random number generator.
    hyperparams : dict
        A dictionary containing the hyperparameters for the ElasticNet model.
    train_array_path : str
        The path to the training dataset.
    model_path : str
        The path where the trained model will be saved.

    Methods
    -------
    train_model():
        Trains the ElasticNet model on the training dataset and saves the trained model.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the ModelTrainer object.
        """
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).model_trainer
        self.params = read_yaml(PARAMS).elasticnet

        # Model Parameters
        self.random_seed = self.params.random_seed
        self.hyperparams = self.params.hyperparameters

        # Input file path
        self.train_array_path = normpath(self.configs.train_array_path)

        # Output file path
        self.model_path = normpath(self.configs.model_path)

    def train_model(self) -> ElasticNet:
        """
        Trains the ElasticNet model on the training dataset and saves the trained model.

        Returns:
            ElasticNet: The trained ElasticNet model.
        """
        try:
            # Load the training set array
            train_array = np.load(self.train_array_path)

            # Split train_array into features and target
            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            # Log the shapes
            logger.info("The shape of x_train: %s", x_train.shape)
            logger.info("The shape of y_train: %s", y_train.shape)

            # Get the hyperparameters
            alpha = self.hyperparams.alpha
            l1_ratio = self.hyperparams.l1_ratio

            # Log the hyperparameters
            logger.info("The hyperparameters used are:\n%s", self.hyperparams)

            # Prepare the model
            en_model = ElasticNet(
                alpha=alpha, l1_ratio=l1_ratio, random_state=self.random_seed
            )
            logger.info("ElasticNet model prepared")

            # Fit the model on training dataset
            en_model.fit(x_train, y_train)
            logger.info("ElasticNet model fitted on training set")

            # Create directory if not exist
            create_directories([dirname(self.model_path)])

            # Saving the preprocessor object
            save_as_joblib(self.model_path, en_model)

            return en_model
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e
