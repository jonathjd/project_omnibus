import logging
import os
from datetime import datetime


def setup_logging():
    os.makedirs(".logs", exist_ok=True)

    log_filename = f".logs/{datetime.now().strftime('%Y-%m-%d_%H-%M')}-transfer.log"

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
