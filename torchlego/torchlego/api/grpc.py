"""gRPC API"""
import logging
from concurrent import futures
import os

import grpc

from torchlego.proto import torchlego_pb2_grpc, torchlego_pb2
from torchlego.core import get_models, run_executable


class TorchLegoService(torchlego_pb2_grpc.TorchLegoServicer):
    """TorchLego gRPC Service"""

    def Models(self, request, context):  # pylint: disable=unused-argument
        """Get Models"""
        return torchlego_pb2.ModelsResponse(models=str(get_models()))

    def Prediction(self, request, context):  # pylint: disable=unused-argument
        """Model Inference Prediction"""
        model_name = request.modelName
        try:
            result = run_executable(model_name, request)
            return torchlego_pb2.PredictionResponse(prediction=result)
        except Exception as exp:  # pylint: disable=broad-except
            logging.error("error running model: %s error: %s",
                          model_name, str(exp))
            if str(exp) == "model not found":
                return torchlego_pb2.PredictionResponse(error=f"model '{model_name}' does not exist")
            if str(exp) == "model not initialized":
                return torchlego_pb2.PredictionResponse(error=f"model '{model_name}' is not initialized")
        return torchlego_pb2.PredictionResponse(error="error running executable")


def init_api_grpc() -> None:
    """start gRPC API"""
    port = int(os.getenv("GRPC_PORT", "8081"))
    logging.info('starting gRPC server on: %d', port)
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=int(os.getenv("GRPC_WORKERS", "1"))),
    )
    torchlego_pb2_grpc.add_TorchLegoServicer_to_server(
        TorchLegoService(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()
