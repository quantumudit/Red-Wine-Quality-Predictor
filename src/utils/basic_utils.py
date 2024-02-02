"""
This module provides utility functions for handling files and directories. It includes
functions for reading YAML files, creating directories, saving data as JSON or
joblib files, and calculating the size of a file or directory. The functions are
designed to handle exceptions and log relevant information for debugging purposes.
"""
import json
from os import makedirs
from os.path import dirname, getsize, normpath
from typing import Any

import joblib
import yaml
from box import Box

from src.exception import CustomException
from src.logger import logger


def read_yaml(yaml_path: str) -> Box:
    """
    This function reads a YAML file from the provided path and returns
    its content as a Box object.

    Args:
        yaml_path (str): The path to the YAML file to be read.

    Raises:
        CustomException: If there is any error while reading the file or
        loading its content, a CustomException is raised with the original
        exception as its argument.

    Returns:
        Box: The content of the YAML file, loaded into a Box object for
        easy access and manipulation.
    """
    try:
        yaml_path = normpath(yaml_path)
        with open(yaml_path, "r", encoding="utf-8") as yf:
            content = Box(yaml.safe_load(yf))
            logger.info("yaml file: %s loaded successfully", yaml_path)
            return content
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e


def create_directories(dir_paths: list, verbose=True) -> None:
    """This function creates directories at the specified paths.

    Args:
        dir_paths (list): A list of directory paths where directories need
        to be created.
        verbose (bool, optional): If set to True, the function will log a message
        for each directory it creates. Defaults to True.
    """
    for path in dir_paths:
        makedirs(normpath(path), exist_ok=True)
        if verbose:
            logger.info("created directory at: %s", path)


def save_as_json(file_path: str, data: dict) -> None:
    """
    This function saves a dictionary as a JSON file at the specified file path.

    Args:
        file_path (str): The path where the JSON file will be saved. If the directories
        in the path do not exist, they will be created.
        data (dict): The dictionary that will be saved as a JSON file.

    Raises:
        CustomException: If there is an error during the file writing process,
        a CustomException will be raised with the original exception as its argument.
    """
    save_path = normpath(file_path)
    makedirs(dirname(save_path), exist_ok=True)
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logger.info("json file saved at: %s", save_path)
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e


def save_as_joblib(file_path: str, serialized_object: Any) -> None:
    """
    Save a serialized object using joblib.

    Args:
        file_path (str): The file path where the serialized object will be saved.
        serialized_object (Any): The object to be serialized and saved.

    Raises:
        CustomException: If there is an error during the saving process.
    """
    save_path = normpath(file_path)
    makedirs(dirname(save_path), exist_ok=True)
    try:
        joblib.dump(serialized_object, save_path)
        logger.info("object saved at: %s", save_path)
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e


def load_joblib(file_path: str) -> joblib:
    """
    This function loads a joblib file from a specified file path.

    Args:
        file_path (str): The path to the joblib file to be loaded.

    Raises:
        CustomException: If there is an error in loading the joblib file,
        a custom exception is raised with the error message.

    Returns:
        joblib: The loaded joblib object
    """
    saved_path = normpath(file_path)
    try:
        joblib_object = joblib.load(saved_path)
        logger.info("object loaded from: %s", saved_path)
        return joblib_object
    except Exception as e:
        logger.error(CustomException(e))
        raise CustomException(e) from e


def get_size(path: str) -> str:
    """
    This function calculates the size of the file or directory at the given path.

    Args:
        path (str): The path of the file or directory for which the size i
        to be calculated.

    Returns:
        str: The size of the file or directory in kilobytes, rounded to the
        nearest kilobyte.
    """
    norm_path = normpath(path)
    size_in_kb = round(getsize(norm_path) / 1024)
    return f"~ {size_in_kb} KB"
