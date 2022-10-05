"""PyTorch Serving gRPC Client"""
import json
import sys

import grpc
import inference_pb2
from inference_pb2_grpc import InferenceStub


def run():
    """Driver program"""
    args = sys.argv
    url = 'localhost:8000'
    if len(args) > 1:
        url = args[1]
    with grpc.insecure_channel(url) as channel:
        stub = InferenceStub(channel)
        response = stub.Health(inference_pb2.google_dot_protobuf_dot_empty__pb2.Empty())
        print("health: " + response.status)
        with open('./model/example.jpg', 'rb') as reader:
            img_bytes = reader.read()
        response = stub.Prediction(inference_pb2.PredictionRequest(input=img_bytes))
        print("prediction: " + json.dumps(json.loads(response.prediction.decode('utf-8'))))

if __name__ == '__main__':
    run()
