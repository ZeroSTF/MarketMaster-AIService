import logging
from colorlog import ColoredFormatter
from ..config.settings import Config


def setup_logger():
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))  # Set log level

    # Define a colored formatter
    formatter = ColoredFormatter(
        "%(log_color)s" + Config.LOG_FORMAT + "%(reset)s",  # Apply colors
        log_colors={
            'DEBUG': 'cyan',       # Cyan for DEBUG
            'INFO': 'green',       # Green for INFO
            'WARNING': 'yellow',   # Yellow for WARNING
            'ERROR': 'red',        # Red for ERROR
            'CRITICAL': 'bold_red' # Bold Red for CRITICAL
        }
    )

    # Create a console handler with the colored formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.hasHandlers():  # Prevent adding multiple handlers
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()
