<h3 align="center">
  <img src="torchlego.png" width="50%">
  <br />
  Model Serving as Code for PyTorch
</h3>

## Getting Started

TorchLego is a server for running inference on PyTorch models. It is inspired by the concept of
X-as-Code e.g. Infrastructure-as-Code, Security-as-Code, etc. With TorchLego, one can define the
preprocess, postprocess and PyTorch TorchScript module location as a config for execution.

### Usage

- Create a TorchLego model configuration file `models.yaml` in YAML format. You can refer to following YAML:

```yaml
models:
  - name: torchvision-resnet50 <- unique name/slug for the model
    download: http://download-link <- module download link
    gpu: true
    stages:
      input: file <- support for file upload as input while running inference
      preprocess:
        default: image_classification <- default torchvision transforms for preprocessing
  - name: custom-resnet50
    download: http://download-link <- module download link
    gpu: true
    stages:
      input: file
      # custom pytorch transforms for preprocessing the input
      preprocess:
        resize: 299
        center_crop: 299
        to_tensor: true
        normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
        unsqueeze: 0
```

- Run the following command to run an [example configuration](examples/). Note, TorchLego picks
  up config files from `model-config` folder from the root directory.

```bash
docker run --rm --net=host -v ${PWD}/examples:/model-config torchlego:latest
```

## Contributing

See the [Contributing Guide](CONTRIBUTNG.md) on how to help out.

## License

This project is licensed under [BSD 3-Clause License](LICENSE).
