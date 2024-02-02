"""
WIP
"""

import sys


def error_details(error):
    """_summary_

    Args:
        error (_type_): _description_

    Returns:
        _type_: _description_
    """
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        "Error occurred in Python script "
        f"[{file_name}] at line [{line_number}]: [{str(error)}]")
    return error_message


class CustomException(Exception):
    """_summary_

    Args:
        Exception (_type_): _description_
    """

    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = error_details(error_message)

    def __str__(self):
        return self.error_message
