import sys
import grpc
import inspect
import time
import gevent
import numpy as np
from PIL import Image
from locust.contrib.fasthttp import FastHttpUser
from locust import User, task, between
from locust.runners import STATE_STOPPING, STATE_STOPPED, STATE_CLEANUP, WorkerRunner
from torchvision import transforms
import tritonclient.grpc as grpcclient

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
    return input

class GrpcUser(User):
    abstract = True

    def __init__(self, *args, **kwargs):
        super(GrpcUser, self).__init__(*args, **kwargs)
        target = self.host.lstrip('http://')
        self.client = grpcclient.InferenceServerClient(url=target)


class ApiUser(GrpcUser):
    wait_time = between(0.9, 1.1)

    @task
    def get_prediction(self):
        input = get_input()
        inputs = grpcclient.InferInput('input__0', input.shape, datatype='FP32')
        inputs.set_data_from_numpy(input)
        outputs = grpcclient.InferRequestedOutput('output__0', class_count=1000)
        start_time = time.time()
        try:
            results = self.client.infer(model_name='efficientnet_b0', inputs=[inputs], outputs=[outputs])
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
            inference_output = results.as_numpy('output__0')
            print(inference_output[:5])
