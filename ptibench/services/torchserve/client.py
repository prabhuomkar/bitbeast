import numpy as np
from torchvision import transforms
from PIL import Image
import grpc
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

def get_result_torchserve():
    input, shape = get_input()
    channel = grpc.insecure_channel('localhost:7070')
    stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    input_data = {'data': input, 'shape': shape}
    response = stub.Predictions(
        inference_pb2.PredictionsRequest(model_name="efficientnet_b0",
                                         input=input_data))
    try:
        prediction = response.prediction.decode('utf-8')
        print(prediction)
    except Exception as e:
        print('inference error', str(e))

if __name__ == '__main__':
    get_result_torchserve()
