"""TorchLego Semantic Segmentation Example"""
import sys
from typing import Dict, List

from PIL import Image
import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights


FILE_NAME = 'LRASPP_MobileNet_V3.pt'


class SemanticSegmentationModule(torch.nn.Module):
    """Semantic Segmentation TorchScript Module"""

    def __init__(self) -> None:
        super().__init__()
        self.model = lraspp_mobilenet_v3_large(
            weights=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
        self.model.eval()

        self.categories = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.meta[
            "categories"]

    def forward(self, img_tensor, categories: List[str]):
        """Forward Pass"""
        output = self.model(img_tensor)["out"]
        probabilties = output.softmax(dim=1)
        result: Dict[str, List[List[float]]] = {}
        for cls in categories:
            idx: int = self.categories.index(cls)
            prob: List[List[float]] = probabilties[0, idx].tolist()
            result[cls] = prob
        return result


def script_and_save():
    """Initialize pytorch model with weights, script it and save the torchscript module"""
    print('scripting and saving torchscript module')
    scripted_module = torch.jit.script(SemanticSegmentationModule())
    scripted_module.save(FILE_NAME)


def load_and_run():
    """Loads the saved torchscript module and runs sample image"""
    print('loading and running torchscript module')
    model = torch.jit.load(FILE_NAME)
    weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    preprocess = weights.transforms()
    img = Image.open('example.jpg')
    img_transformed = preprocess(img)
    print(model(img_transformed.unsqueeze(0), ['diningtable']))


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        if args[1] == 'save':
            script_and_save()
            exit(0)
        if args[1] == 'run':
            load_and_run()
            exit(0)
    print('provide a valid arg: save OR run')
