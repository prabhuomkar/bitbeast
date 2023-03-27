# golibtorch
PyTorch Model Inference in Golang

## Directory Structure

- [model](model) - example torchscript module with example image and class to label mapping file
- [golibtorch](golibtorch) - C++ header and source files with CGO wrapper
- [main.go](main.go) - Driver program which executes the CGO wrapper with model and example input

## Getting Started 

### Downloading LibTorch (PyTorch C++ API)
Run following command to download and setup LibTorch (CPU version):
```
make deps
```

### Generating TorchScript Module
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

### Run Example
Run the program with sample input:
```bash
make run
```

## License
This project is licensed under [MIT License](LICENSE).