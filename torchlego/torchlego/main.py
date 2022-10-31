"""TorchLego"""
import os
import logging
import logging.config

from config import LOGGING_CONFIG
from core import initialize_models
from api import start_http

logging.config.dictConfig(LOGGING_CONFIG)

if __name__ == '__main__':
    initialize_models()
    start_http(int(os.getenv("HTTP_PORT", 8080)))
