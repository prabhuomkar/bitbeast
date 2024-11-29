"""PyTorch Serving Example Model"""
import sys
from typing import Dict, Any

from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


FILE_NAME = 'MobileNet_V3_Small_IMAGENET1K_V1.pt'

class MobileNetV3SmallModule(torch.nn.Module):
    """EfficientNet TorchScript Module"""
    def __init__(self) -> None:
        super(MobileNetV3SmallModule, self).__init__()

        # load the pretrained model
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.model.eval()

        # map imagenet classes to labels
        self.categories = [s.strip() for s in \
            open('imagenet_classes.txt', encoding='utf-8').readlines()]

    def forward(self, img_tensor: torch.Tensor, topk: int=5) -> Dict[str, Any]:
        """Forward Pass"""
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_class = torch.topk(probabilities, topk)
        # return {"class": "probability score"} e.g. {"pizza": 0.44922224}
        return dict({self.categories[top_class[0][idx].item()]: top_prob[0][idx].item() \
            for idx in range(0, int(topk))})

def script_and_save():
    """Initialize pytorch model with weights, script it and save the torchscript module"""
    print('scripting and saving torchscript module')
    scripted_module = torch.jit.script(MobileNetV3SmallModule())
    scripted_module.save(FILE_NAME)

def load_and_run():
    """Loads the saved torchscript module and runs sample image"""
    print('loading and running torchscript module')
    model = torch.jit.load(FILE_NAME)
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    img = Image.open('example.jpg')
    img_transformed = preprocess(img)
    print(model(img_transformed.unsqueeze(0)))

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
