"""TorchLego"""
from dotenv import load_dotenv
load_dotenv("model-config/.env")


if __name__ == '__main__':
    from api import init_api_http
    from core import init_models
    from config import init_logging

    init_logging()
    init_models()
    init_api_http()
