import logging
from ..config.settings import Config

def setup_logger():
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=Config.LOG_FORMAT
    )
    return logging.getLogger(__name__)

logger = setup_logger()