# Notes

## Roadmap

- [x] Configurable HTTP/gRPC API
- [x] YAML Configuration: Preprocess, Inference
- [x] Computer Vision Tasks
- [x] Release Docker Image (cpu version)
- [x] GPU support
- [ ] Error Handling
- [ ] NLP Tasks
- [ ] Release Docker Image (gpu version)

## YAML Example

```
models:
  - name: torchvision-resnet50
    download: https://artifactory-link/model/version.pt
    gpu: true
    precision: half
    stages:
      input: file
      preprocess:
        default: image_classification
  - name: custom-resnet50
    download: https://artifactory-link/model/version.pt
    gpu: true
    precision: default
    stages:
      input: file
      preprocess:
        resize: 299
        center_crop: 299
        to_tensor: true
        normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
        unsqueeze: 0
```
