"""REST API"""
import logging

from sanic import Sanic, response

from config import LOGGING_CONFIG
from core import get_models, run_executable

async def models(request): # pylint: disable=unused-argument
    """Get Models"""
    return response.json(status=200, body=get_models())

async def prediction(request, model_name: str): # pylint: disable=unused-argument
    """Model Inference Prediction"""
    try:
        result = run_executable(model_name, request)
        return response.json(status=200, body=result)
    except Exception as exp: # pylint: disable=broad-except
        logging.error("error running model: %s error: %s", model_name, str(exp))
        if str(exp) == "model not found":
            return response.json(status=404, body={"message": f"model '{model_name}' does not exist"})
        if str(exp) == "model not initialized":
            return response.json(status=403, body={"message": f"model '{model_name}' is not initialized"})
    return response.json(status=500, body={"message": "error running executable"})

def add_routes(app: Sanic) -> None:
    """Add routes with handlers"""
    app.add_route(models, "/v1/models", methods=["GET"])
    app.add_route(prediction, "/v1/models/<model_name:str>", methods=["POST"])

def start_http(port: int) -> None:
    """start REST API"""
    logging.info('starting HTTP server on: %d', port)
    app = Sanic('TorchLego', log_config=LOGGING_CONFIG)
    add_routes(app)
    app.run(host='0.0.0.0', port=port, debug=False, access_log=False)
