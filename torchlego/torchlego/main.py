"""TorchLego"""
from dotenv import load_dotenv
load_dotenv("model-config/.env")


if __name__ == '__main__':
    from torchlego.api import init_api_http, init_api_grpc
    from torchlego.core import init_models
    from torchlego.config import init_logging

    init_logging()
    init_models()
    init_api_grpc()
    init_api_http()
