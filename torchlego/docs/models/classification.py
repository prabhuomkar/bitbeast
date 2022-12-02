"""TorchLego Image Classification Example"""
import sys

from PIL import Image
import torch
from torchvision import transforms
from timm import create_model


FILE_NAME = 'MobileNet_V3.pt'


class ImageClassificationModule(torch.nn.Module):
    """Image Classification TorchScript Module"""

    def __init__(self) -> None:
        super().__init__()
        self.model = create_model(
            'mobilenetv2_100', pretrained=True, scriptable=True)
        self.model.eval()

        self.categories = [s.strip() for s in
                           open('imagenet_classes.txt', encoding='utf-8').readlines()]

    def forward(self, img_tensor, topk: int = 5):
        """Forward Pass"""
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_class = torch.topk(probabilities, topk)
        return dict({self.categories[top_class[0][idx].item()]: top_prob[0][idx].item()
                     for idx in range(0, int(topk))})


def script_and_save():
    """Initialize pytorch model with weights, script it and save the torchscript module"""
    print('scripting and saving torchscript module')
    scripted_module = torch.jit.script(ImageClassificationModule())
    scripted_module.save(FILE_NAME)


def load_and_run():
    """Loads the saved torchscript module and runs sample image"""
    print('loading and running torchscript module')
    model = torch.jit.load(FILE_NAME)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
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
