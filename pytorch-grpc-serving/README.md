# pytorch-grpc-serving
Serving PyTorch Models for Inference as gRPC API

## Directory Structure

- [model](model) - example torchscript module with example image and class to label mapping file
- [protos](protos) - protocol buffer defintion for gRPC service
- [client.py](client.py) - gRPC client example for connecting to gRPC server to run inference on sample file in [model](model) directory
- [Dockerfile](Dockerfile) - containerize the gRPC server to for deployment
- [Makefile](Makefile) - commands for generating stubs, running gRPC service, building docker image, other chores
- [requirements.txt](requirements.txt) - python requirements for running this service
- [server.py](server.py) - gRPC server for loading and running model inference

## Getting Started 

### Releasing Model
- Refer [model/torchscript.py](model/torchscript.py) for conversion of torchvision model to TorchScript module
- TorchScript module is written as follows:
```python
class YourModule(nn.Module):
    def __init__(self):
        # initialize the quantized model with pretrained weights
        # load class to label dictionary
    def forward(self, input):
        # run forward pass and compute classes with its probabilities
        # map classes to labels
        # return result
```
- The example file uses [ImageNet Classes](model/imagenet_classes.txt) for mapping imagenet class to its human readable label
- Run following command to create TorchScript module:
```bash
cd model
python torchscript.py save
```
- Run following command to load TorchScript module and run inference over sample image:
```bash
cd model
python torchscript.py run
```

### Setting up gRPC Service
- Refer [protos/inference.proto](protos/inference.proto) for Protocol Buffer definition of the gRPC Service
- Run following command to generate the stubs:
```bash
make proto
```

### Running gRPC Server
- Run the following command to start the gRPC Server
```bash
python server.py
```

### Running gRPC Client
- Run the following command to start the gRPC Client
```bash
python client.py <url>
# python client.py localhost:8000
```

## Example
- Example server with quantized MobileNet V3 Small is hosted on [Fly](https://fly.io).
- Configure client to run inferencing on the hosted server:
```
python client.py pytorch-serving.fly.dev:8000
```
- Example Output:
```
health: UP
prediction: {"pizza": 0.44644981622695923, "potpie": 0.009865873493254185, "hot pot": 0.007180684246122837, "consomme": 0.005226321052759886, "spatula": 0.0047011710703372955}
```

## License
This project is licensed under [MIT License](LICENSE).