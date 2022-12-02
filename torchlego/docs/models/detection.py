"""TorchLego Object Detection Example"""
import sys
from typing import Dict, Union, List

from PIL import Image
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights


FILE_NAME = 'SSDLite320_MobileNet_V3.pt'


class ObjectDetectionModule(torch.nn.Module):
    """Object Detection TorchScript Module"""

    def __init__(self) -> None:
        super().__init__()
        self.model = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
        self.model.eval()

        self.categories = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.meta["categories"]

    def forward(self, img_tensor, min_score: float = 0.8):
        """Forward Pass"""
        output = self.model([img_tensor])[1][0]
        result: List[Dict[str, Union[List[float], float, str]]] = []
        for idx, score in enumerate(output['scores']):
            if float(score.item()) >= min_score:
                box: List[float] = output['boxes'][idx].tolist()
                result.append(dict({
                    'box': box,
                    'score': float(score.item()),
                    'label': self.categories[output['labels'][idx]],
                }))
        return result


def script_and_save():
    """Initialize pytorch model with weights, script it and save the torchscript module"""
    print('scripting and saving torchscript module')
    scripted_module = torch.jit.script(ObjectDetectionModule())
    scripted_module.save(FILE_NAME)


def load_and_run():
    """Loads the saved torchscript module and runs sample image"""
    print('loading and running torchscript module')
    model = torch.jit.load(FILE_NAME)
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    preprocess = weights.transforms()
    img = Image.open('example.jpg')
    img_transformed = preprocess(img)
    print(model(img_transformed))


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
