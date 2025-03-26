import logging
from datetime import datetime
import os


def setup_logging():
    # Ensure .logs directory exists
    os.makedirs(".logs", exist_ok=True)

    # Create log filename with timestamp down to seconds
    log_filename = f".logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-transfer.log"

    # create logger
    logger = logging.getLogger("transfer_pipeline")
    logger.setLevel(logging.INFO)

    # formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to root logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
