from torchvision import transforms
from PIL import Image
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
    return preprocess(img).numpy()

# nvidia triton inference server
def get_result_triton():
    input = get_input()
    client = triton_client = grpcclient.InferenceServerClient(url='localhost:8001')
    inputs = grpcclient.InferInput('input__0', input.shape, datatype='FP32')
    inputs.set_data_from_numpy(input)
    outputs = grpcclient.InferRequestedOutput('output__0', class_count=1000)
    try:
        results = client.infer(model_name='efficientnet_b0', inputs=[inputs], outputs=[outputs])
        inference_output = results.as_numpy('output__0')
        print(inference_output[:5])
    except Exception as e:
        print('inference error', str(e))

if __name__ == '__main__':
    get_result_triton()
