"""TorchLego"""
from dotenv import load_dotenv
load_dotenv()

from config import init_logging
from core import init_models
from api import init_api_http

if __name__ == '__main__':
    init_logging()
    init_models()
    init_api_http()
