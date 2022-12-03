# TorchLego Documentation: Tasks

## Computer Vision

Supported Libraries:

- [TorchVision](https://pytorch.org/vision/stable/models.html#)
- [TIMM](https://huggingface.co/docs/timm/index)

### Image Classification

- [MobileNetV3 from TIMM](https://huggingface.co/docs/timm/models#mobilenetv3)
- [Convert Model to TorchScript Module](models/classification.py)
- Model Configuration:
```yaml
models:
  - name: mobilenetv3
    download: https://www.dropbox.com/s/wud1np4y8r1zkqh/MobileNet_V3.pt?dl=1
    stages:
      input: file
      preprocess: 
        default: image_classification
```

### Semantic Segmenation

- [LRASPP MobileNetV3 from TorchVision](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html#torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights)
- [Convert Model to TorchScript Module](models/segmentation.py)
- Model Configuration:
```yaml
models:
  - name: lraspp-mobilenetv3
    download: https://www.dropbox.com/s/2eb02iyd8cd9qe7/LRASPP_MobileNet_V3.pt?dl=1
    stages:
      input: file
      preprocess: 
        default: semantic_segmentation
```

### Object Detection

- [SSDLite MobileNetV3 from TorchVision](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html#torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights)
- [Convert Model to TorchScript Module](models/detection.py)
```yaml
models:
  - name: ssdlite-mobilenetv3
    download: https://www.dropbox.com/s/5hk7p8o5y092njc/SSDLite320_MobileNet_V3.pt?dl=1
    stages:
      input: file
      preprocess: 
        default: object_detection
```

## Natural Language Processing

Supported Libraries:

- [HuggingFace](https://huggingface.co/docs/transformers/index)
- [SBERT](https://www.sbert.net/)

### Sentiment Classification

TBD models

### Text Classification

TBD models

### Question Answering

TBD models
