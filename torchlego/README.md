# TorchLego

Model Serving as Code For PyTorch

## Getting Started

TorchLego is a server for running inference on PyTorch models. It is inspired by the concept of X-as-Code e.g. Infrastructure-as-Code, Security-as-Code, etc. With TorchLego, one can define the preprocess, postprocess and PyTorch TorchScript module location as a config for execution.

### Usage

Run the following command to run an [example configuration](examples/).

```
docker run --rm --net=host -v ${PWD}/examples:/ torchlego:latest
```

## Contributing

See the [Contributing Guide](CONTRIBUTNG.md) on how to help out.

## License

This project is licensed under [BSD 3-Clause License](LICENSE).
