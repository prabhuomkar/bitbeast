# Changelog
All notable changes to this project will be documented in this file.  
This project adheres to [Calendar Versioning](https://calver.org/).

## [2023.01.19](https://hub.docker.com/layers/prabhuomkar/torchlego/2023.01.19/images/sha256-80f723944c65998f7b20f8e6bcb4f4aa6a471665ffb304bf54057ab1c21bc928?context=explore)

- Support for setting input precision during inference. Set `precision: default` or `precision: half` for using it.
- Fixed CUDA support, Docker image still supports CPU though.

## [2023.01.15](https://hub.docker.com/layers/prabhuomkar/torchlego/2023.01.15/images/sha256-ace62cf746417c65fb38031a3f6e28043d54744356bf8d13e67f0d73a38a225b?context=explore)

- Support for using CUDA device for input and inference. Set `gpu: true` for using it.

## [2022.12.03](https://hub.docker.com/layers/prabhuomkar/torchlego/2022.12.03/images/sha256-ddce83c174f24038105172599414b68d41a6738a594c7ae89b496ebd0897b05c?context=explore)

- Task Documentation with Computer Vision [Examples](docs/TASKS.md) from TIMM & TorchVision.
- Fixed `preprocessing` stage to work with both default and custom steps.
- Made `gpu` as optional parameter.

## [2022.11.29](https://hub.docker.com/layers/prabhuomkar/torchlego/2022.11.29/images/sha256-86fad4a49daf58f8b0fcb3a0b3ccba86d7c4989f37a9b53239eba03a5ccb95f7?context=explore)

- Initial release with basic steps for image classification task.
- [Examples](examples) for running server with quantized model.