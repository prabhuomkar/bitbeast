"""TorchLego Logging Config"""
import os


LOGGING_CONFIG = {
    "version": 1,
    "root": {
        "level": os.getenv("LOG_LEVEL", "INFO")
    }
}
