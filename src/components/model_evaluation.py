"""
wip
"""
import os
from os.path import basename, join, normpath
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np
from dotenv import load_dotenv

from src.constants import CONFIGS, PARAMS
from src.exception import CustomException
from src.logger import logger
from src.utils.basic_utils import create_directories, load_joblib, read_yaml
from src.utils.model_utils import log_scores, regression_metrics

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelEvaluation:
    """_summary_"""

    def __init__(self):
        """_summary_"""
        # Read the configuration files
        self.configs = read_yaml(CONFIGS).model_evaluation
        self.params = read_yaml(PARAMS).elasticnet

        # Input file path
        self.train_array_path = normpath(self.configs.train_array_path)
        self.test_array_path = normpath(self.configs.test_array_path)
        self.model_path = normpath(self.configs.model_path)

        # Output file path
        self.scores_dir = normpath(self.configs.scores_dir)
        self.preds_dir = normpath(self.configs.predictions_dir)

    def get_features_and_labels(self) -> tuple[np.array]:
        """_summary_

        Args:
            self (_type_): _description_

        Raises:
            CustomException: _description_

        Returns:
            _type_: _description_
        """
        try:
            # Load the training & test set array
            train_array = np.load(self.train_array_path)
            test_array = np.load(self.test_array_path)

            # Split train_array into features and target
            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Log the shapes
            logger.info("The shape of x_train: %s", x_train.shape)
            logger.info("The shape of y_train: %s", y_train.shape)
            logger.info("The shape of x_test: %s", x_test.shape)
            logger.info("The shape of y_test: %s", y_test.shape)
            return (x_train, y_train, x_test, y_test)
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e

    @staticmethod
    def generate_filename(model_filepath: str, suffix: str, extension: str) -> str:
        """_summary_

        Args:
            model_filepath (str): _description_
            suffix (str): _description_
            extension (str): _description_

        Returns:
            str: _description_
        """
        model_name = basename(model_filepath).split(".")[0]
        if "." in extension:
            return model_name + suffix + extension
        return model_name + suffix + "." + extension

    def get_predictions(self) -> tuple[np.array]:
        """_summary_

        Args:
            self (_type_): _description_

        Raises:
            CustomException: _description_

        Returns:
            _type_: _description_
        """

        try:
            # Load the model
            en_model = load_joblib(self.model_path)

            # load train and test features
            x_train, _, x_test, _ = self.get_features_and_labels()

            # Perform predictions
            y_train_preds = en_model.predict(x_train)
            y_test_preds = en_model.predict(x_test)
            logger.info("predictions on training and test data completed")

            # Log the shape
            logger.info("Shape of y_train_preds:%s", {y_train_preds.shape})
            logger.info("Shape of y_test_preds:%s", {y_test_preds.shape})

            return (y_train_preds, y_test_preds, en_model)
        except Exception as e:
            logger.error(CustomException(e))
            raise CustomException(e) from e

    def evaluate_model(self) -> dict:
        """_summary_

        Raises:
            CustomException: _description_

        Returns:
            dict: _description_
        """
        try:
            # load train and test labels
            x_train, y_train, x_test, y_test = self.get_features_and_labels()

            # load train and test predictions
            y_train_preds, y_test_preds, en_model = self.get_predictions()

            # Get hyperparameters for the model
            hyperparameters = self.params.hyperparameters

            # Evaluate model
            train_eval_metrics = regression_metrics(
                y_train, y_train_preds, x_train.shape
            )
            test_eval_metrics = regression_metrics(y_test, y_test_preds, x_test.shape)

            # Additional Model info:
            model_info = {
                "estimator_type": en_model._estimator_type,
                "coefficients": en_model.coef_.tolist(),
                "intercept": en_model.intercept_,
                "dual_gap": en_model.dual_gap_,
                "input_features_count": en_model.n_features_in_,
                "iteration_count": en_model.n_iter_,
                "all_params": en_model.get_params(),
            }

            return {
                "train_eval_metrics": train_eval_metrics,
                "test_eval_metrics": test_eval_metrics,
                "y_train_preds": y_train_preds,
                "y_test_preds": y_test_preds,
                "en_model": en_model,
                "model_info": model_info,
                "hyperparameters": hyperparameters,
            }
        except Exception as e:
            logger.info(CustomException(e))
            raise CustomException(e) from e

    def log_into_mlflow(self):
        """_summary_

        Raises:
            CustomException: _description_
        """
        eval_details = self.evaluate_model()
        try:
            hyperparameters = eval_details.get("hyperparameters")
            test_eval_metrics = eval_details.get("test_eval_metrics")
            en_model = eval_details.get("en_model")

            logger.info("Started logging information to MLFlow")

            mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                mlflow.log_params(hyperparameters)
                mlflow.log_metrics(test_eval_metrics)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        en_model,
                        "model",
                        registered_model_name="ElasticNetModel",
                    )
                else:
                    mlflow.sklearn.log_model(en_model, "model")
                logger.info("MLFlow logging completed")
        except Exception as e:
            logger.info(CustomException(e))
            raise CustomException(e) from e

    def save_evaluation_results(self):
        """_summary_"""
        eval_details = self.evaluate_model()
        hyperparameters = eval_details.get("hyperparameters")
        train_eval_metrics = eval_details.get("train_eval_metrics")
        test_eval_metrics = eval_details.get("test_eval_metrics")
        y_train_preds = eval_details.get("y_train_preds")
        y_test_preds = eval_details.get("y_test_preds")
        model_info = eval_details.get("model_info")
        model_name = "ElasticNet"

        # Create directory to save predictions & model score
        create_directories([self.preds_dir, self.scores_dir])

        # Generate training prediction file name & save path
        y_train_preds_filename = self.generate_filename(
            self.model_path, "_train_preds_arr", "npy"
        )
        y_train_preds_filepath = normpath(join(self.preds_dir, y_train_preds_filename))

        # Generate test prediction file name & save path
        y_test_preds_filename = self.generate_filename(
            self.model_path, "_test_preds_arr", "npy"
        )
        y_test_preds_filepath = normpath(join(self.preds_dir, y_test_preds_filename))

        # save the training predictions
        np.save(y_train_preds_filepath, y_train_preds)
        logger.info("training predictions saved at: %s", y_train_preds_filepath)

        # save the training predictions
        np.save(y_test_preds_filepath, y_test_preds)
        logger.info("test predictions saved at: %s", y_test_preds_filepath)

        # Create filename and filepath to log scores
        scores_filename = self.generate_filename(self.model_path, "_scores", "json")
        scores_filepath = normpath(join(self.scores_dir, scores_filename))

        # Save the model scores
        log_scores(
            scores_filepath,
            hyperparameters,
            train_eval_metrics,
            test_eval_metrics,
            model_name,
            **model_info
        )

        logger.info("Scores recorded in: %s", scores_filepath)
