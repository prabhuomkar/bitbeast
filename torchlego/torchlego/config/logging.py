"""TorchLego Logging Config"""
import logging
import os

LOGGING_CONFIG = {
    "version": 1,
    "root": {
        "level": os.getenv("LOG_LEVEL", "INFO")
    }
}

def init_logging() -> None:
    """Initialize logging"""
    logging.config.dictConfig(LOGGING_CONFIG)
