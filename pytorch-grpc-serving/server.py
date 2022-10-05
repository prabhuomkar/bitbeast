"""PyTorch Serving gRPC Server"""
from concurrent import futures
import io
import json

import grpc
from PIL import Image
from torchvision.models.quantization import ResNet50_QuantizedWeights
from torch.jit import load
import inference_pb2
from inference_pb2_grpc import InferenceServicer, add_InferenceServicer_to_server


class PyTorchInferenceServicer(InferenceServicer):
    """PyTorch Inference Service"""
    # instance to access torchscript module
    model = None
    def __init__(self) -> None:
        # load model
        try:
            self.model = load('./model/ResNet50_Quantized_IMAGENET1K_FBGEMM_V2.pt')
            self.weights = ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V2
            self.transforms = self.weights.transforms()
        except:
            print('some error loading model')

    def Health(self, request, context):
        if self.model is not None and self.model.parameters() is not None:
            return inference_pb2.HealthResponse(status="UP")
        return inference_pb2.HealthResponse(status="DOWN")

    def Prediction(self, request, context):
        img = Image.open(io.BytesIO(request.input))
        img_transformed = self.transforms(img)
        result = self.model(img_transformed.unsqueeze(0))
        return inference_pb2.PredictionResponse(prediction=json.dumps(result).encode('utf-8'))

if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_InferenceServicer_to_server(PyTorchInferenceServicer(), server)
    server.add_insecure_port('[::]:8000')
    server.start()
    server.wait_for_termination()
