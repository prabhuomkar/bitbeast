"""PyTorch Serving gRPC Server"""
from concurrent import futures
import io
import json

import grpc
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights
from executorch.runtime import Runtime
import torch
import inference_pb2
from inference_pb2_grpc import InferenceServicer, add_InferenceServicer_to_server


class PyTorchInferenceServicer(InferenceServicer):
    """PyTorch Inference Service"""
    # instance to access torchscript module
    model = None
    def __init__(self) -> None:
        # load model
        try:
            self.weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.transforms = self.weights.transforms()
            self.runtime = Runtime.get()
            self.program = self.runtime.load_program('./model/MobileNet_V3_Small_IMAGENET1K_V1.pte')
            self.method = self.program.load_method('forward')
            with open('./model/imagenet_classes.txt', encoding='utf-8') as f:
                self.categories = [s.strip() for s in f.readlines()]
        except:
            print('some error loading model')

    def Health(self, request, context):
        if self.method is not None and self.method.metadata is not None:
            return inference_pb2.HealthResponse(status="UP")
        return inference_pb2.HealthResponse(status="DOWN")

    def Prediction(self, request, context):
        img = Image.open(io.BytesIO(request.input))
        img_transformed = self.transforms(img)
        inputs = img_transformed.unsqueeze(0).contiguous()
        outputs = self.method.execute([inputs])
        top_prob, top_class = torch.topk(outputs[0], 5)
        result = dict({self.categories[top_class[0][idx].item()]: top_prob[0][idx].item() \
            for idx in range(0, 5)})
        return inference_pb2.PredictionResponse(prediction=json.dumps(result).encode('utf-8'))

if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_InferenceServicer_to_server(PyTorchInferenceServicer(), server)
    server.add_insecure_port('[::]:8000')
    server.start()
    server.wait_for_termination()
