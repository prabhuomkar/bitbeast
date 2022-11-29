# Notes

## Roadmap

- [x] Configurable HTTP/gRPC API
- [x] YAML Configuration: Preprocess, Inference
- [x] Computer Vision Tasks
- [ ] Error Handling
- [ ] Release Docker Image (cpu/gpu versions as per python)
- [ ] NLP Tasks

## YAML Example

```
models:
  - name: torchvision-resnet50
    download: https://www.dropbox.com/s/tqfc8ou1w3hx4gg/ResNet50_Quantized_IMAGENET1K_FBGEMM_V2.pt?dl=1
    gpu: true
    stages:
      input: file
      preprocess:
        default: image_classification
  - name: custom-resnet50
    download: https://www.dropbox.com/s/tqfc8ou1w3hx4gg/ResNet50_Quantized_IMAGENET1K_FBGEMM_V2.pt?dl=1
    gpu: true
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
