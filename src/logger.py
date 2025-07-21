import os
import logging


from logging.handlers import RotatingFileHandler


def setup_logging():

    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        "logs/application.log", maxBytes=10485760, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    return root_logger


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Logging Has Started")
