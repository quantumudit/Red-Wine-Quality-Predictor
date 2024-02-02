"""wip
"""

import logging
import os
import sys
from datetime import datetime

# Setup logs directory in the current working directory
logs_dir_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir_path, exist_ok=True)

# Setup log file name
timestamp_fmt = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")
logs_filename = f"{timestamp_fmt}.log"

# Create the logs file in the logs directory
LOG_FILE_PATH = os.path.join(logs_dir_path, logs_filename)

# Create logging string
LOGGING_STR = "[%(asctime)s]:%(name)s %(levelname)s: %(module)s %(lineno)d - %(message)s"

# Creates logging configuration
logging.basicConfig(
    level=logging.INFO,
    encoding="utf-8",
    format=LOGGING_STR,
    datefmt="%Y-%m-%d %I:%M:%S %p",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create the logger object
logger = logging.getLogger("ProjectLogger")
