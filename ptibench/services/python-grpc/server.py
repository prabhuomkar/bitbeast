from concurrent import futures
import numpy as np
import json
import grpc
import torch
import inference_pb2
from inference_pb2_grpc import InferenceServicer, add_InferenceServicer_to_server


class PyTorchInferenceServicer(InferenceServicer):
    model = None
    def __init__(self) -> None:
        self.topk = 5
        try:
            self.model = torch.jit.load('./ts_model.pt')
        except:
            print('some error loading model')

    def Prediction(self, request, context):
        input = np.frombuffer(request.input, dtype=np.float32)
        shape = np.frombuffer(request.shape, dtype=int)
        input = torch.reshape(torch.from_numpy(input), tuple(shape))
        result = self.model(input.unsqueeze(0).to('cuda'))
        ps = torch.nn.functional.softmax(result, dim=1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        result = [probs.tolist() + classes.tolist()]
        return inference_pb2.PredictionResponse(prediction=json.dumps(result).encode('utf-8'))

if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_InferenceServicer_to_server(PyTorchInferenceServicer(), server)
    server.add_insecure_port('[::]:8080')
    server.start()
    server.wait_for_termination()
