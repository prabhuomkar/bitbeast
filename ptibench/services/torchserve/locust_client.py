import sys
import grpc
import inspect
import time
import gevent
import numpy as np
from torchvision import transforms
from PIL import Image
from locust.contrib.fasthttp import FastHttpUser
from locust import User, task, between
from locust.runners import STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP, WorkerRunner
import inference_pb2
import inference_pb2_grpc

# preprocessing function
def get_input(img_path='example.jpg'):
    img = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input = preprocess(img).numpy()
    return input.tobytes(), np.asarray(input.shape, dtype=int).tobytes()

class GrpcUser(User):
    abstract = True

    def __init__(self, *args, **kwargs):
        super(GrpcUser, self).__init__(*args, **kwargs)
        target = self.host.lstrip('http://')
        channel = grpc.insecure_channel(target)
        self.stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel=channel)


class ApiUser(GrpcUser):
    wait_time = between(0.9, 1.1)

    @task
    def get_prediction(self):
        input, shape = get_input()
        input_data = {'data': input, 'shape': shape}
        start_time = time.time()
        try:
            response = self.stub.Predictions(inference_pb2.PredictionsRequest(
                                                            model_name="efficientnet_b0",
                                                            input=input_data))
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            print(e)
            self.environment.events.request_failure.fire(request_type="grpc",
                                                         name=self.host,
                                                         response_time=total_time,
                                                         exception=e,
                                                         response_length=0)
        else:
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request_success.fire(request_type="grpc",
                                                         name=self.host,
                                                         response_time=total_time,
                                                         response_length=0)
            prediction = response.prediction.decode('utf-8')
            print(prediction)
