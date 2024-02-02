"""
EDA Helper Functions Module

Author: Udit Kumar Chatterjee
Email: quantumudit@gmail.com

This script contains various helper functions that can be used to perform common tasks
and operations in data analysis.
Functions include:

- dataframe_structure: Returns various attributes associated with the dataframe
- dict_to_table: Generate a pretty looking structure of the output
- datatype_details: Prints the details of the datatype available in the dataframe
- regression_metrics: Calculates various regression metrics.
"""
import pandas as pd
from tabulate import tabulate


def dataframe_structure(dataframe: pd.DataFrame) -> dict:
    """
    This function takes in a Pandas DataFrame as an input and returns a
    dictionary containing details about the structure of the dataframe.
    The returned dictionary includes information such as the number of
    dimensions, shape, row and column count, total and non-null data points,
    total memory usage, and average memory usage of the dataframe.

    Args:
        dataframe (pd.DataFrame): The input DataFrame for which structure details
        are to be extracted

    Returns:
        dict: A dictionary containing the structure details of the input DataFrame
    """
    structure_details = {
        "Dimensions": dataframe.ndim,
        "Shape": dataframe.shape,
        "Row Count": len(dataframe),
        "Column Count": len(dataframe.columns),
        "Total Datapoints": dataframe.size,
        "Null Datapoints": dataframe.isnull().sum().sum(),
        "Non-Null Datapoints": dataframe.notnull().sum().sum(),
        "Total Memory Usage": dataframe.memory_usage(deep=True).sum(),
        "Average Memory Usage": dataframe.memory_usage(deep=True).mean().round(),
    }

    return structure_details


def dict_to_table(input_dict: dict, column_headers: list) -> tabulate:
    """
    This function creates a tabular view of the dictionary results

    Args:
        input_dict (dict): The input dictionary to be pretty printed
        column_headers (list): The list of column headers as a list

    Returns:
        tabulate object: unicode tabular structure of the dataframe
    """
    table_vw = tabulate(
        input_dict.items(), headers=column_headers, tablefmt="pretty", stralign="left"
    )

    return table_vw


def datatype_details(df: pd.DataFrame) -> str:
    """
    This function takes a pandas DataFrame as input and returns a string detailing
    the number of fields for each datatype present in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame for which the datatype details are
        to be determined.

    Returns:
        str: A string message indicating the number of fields for each datatype
        in the input DataFrame.
    """
    available_dtypes = list(set([str(dt) for dt in df.dtypes]))
    for dt in available_dtypes:
        field_count = df.select_dtypes(dt).dtypes.count()
        return f"There are {field_count} fields with {dt} datatype"
