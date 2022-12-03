"""TorchLego Preprocess Stage"""
from typing import Any, Dict
from functools import partial

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms._presets import ImageClassification, SemanticSegmentation, ObjectDetection

DEFAULT_MAPPER = {
    "image_classification": partial(ImageClassification, crop_size=224)(),
    "semantic_segmentation": partial(SemanticSegmentation, resize_size=520)(),
    "object_detection": ObjectDetection()
}

CUSTOM_MAPPER = {
    "resize": Resize,
    "center_crop": CenterCrop,
    "to_tensor": ToTensor,
    "normalize": Normalize,
    "unsqueeze": lambda value: lambda output: torch.unsqueeze(output, value)
}


def defaults(name: str) -> Any:
    """Default preprocessing functions"""
    return DEFAULT_MAPPER.get(name, None)


def to_init(name: str) -> Any:
    """Return if function should be initialized with arguments"""
    if name == "to_tensor":
        return False
    return True


def to_init_with_dict(name: str) -> Any:
    """Return if function should be initialized with dictionary keyword arguments"""
    if name == "normalize":
        return True
    return False


def get_preproces_fn(name: str, value: Any) -> Any:
    """Converts preprocessing stage function with args to executable function"""
    func = CUSTOM_MAPPER.get(name, None)
    if func is not None:
        if not to_init(name):
            return func()
        if to_init_with_dict(name):
            return func(**dict(value))
        return func(value)
    return None


def derive_preprocess(preprocess: Dict[Any, Any]) -> Any:
    """Derive preprocessing stage"""
    # default preprocessing stage
    transforms = []
    if "default" in preprocess:
        transforms.append(defaults(preprocess["default"]))
    del preprocess["default"]
    # custom preprocessing stage
    for name, value in preprocess.items():
        transforms.append(get_preproces_fn(name, value))
    return [Compose(transforms)]
