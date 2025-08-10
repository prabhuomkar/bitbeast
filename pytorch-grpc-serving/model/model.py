"""PyTorch Serving Example Model"""
import sys

from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime


FILE_NAME = 'MobileNet_V3_Small_IMAGENET1K_V1.pte'

class MobileNetV3SmallModule(torch.nn.Module):
    """MobileNet Module"""
    def __init__(self) -> None:
        super(MobileNetV3SmallModule, self).__init__() # pylint: disable=super-with-arguments

        # load the pretrained model
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.model.eval()

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Forward Pass"""
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        return probabilities

def save():
    """Initialize pytorch model with weights and exports the executorch model"""
    print('exporting and saving executorch module')
    model = MobileNetV3SmallModule()
    inputs = (torch.randn(1, 3, 224, 224),)
    et_program = to_edge_transform_and_lower(
        torch.export.export(model, inputs),
        partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    with open(FILE_NAME, 'wb') as f:
        f.write(et_program.buffer)

def run():
    """Loads the executorch model and runs sample image"""
    print('loading and running executorch module')
    runtime = Runtime.get()
    program = runtime.load_program(FILE_NAME)
    method = program.load_method('forward')
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    img = Image.open('example.jpg')
    img_transformed = preprocess(img)
    inputs = img_transformed.unsqueeze(0).contiguous()
    outputs = method.execute([inputs])
    with open('imagenet_classes.txt', encoding='utf-8') as f:
        categories = [s.strip() for s in f.readlines()]
    top_prob, top_class = torch.topk(outputs[0], 5)
    print(dict({categories[top_class[0][idx].item()]: top_prob[0][idx].item() \
            for idx in range(0, 5)}))

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        if args[1] == 'save':
            save()
            sys.exit(0)
        if args[1] == 'run':
            run()
            sys.exit(0)
    print('provide a valid arg: save OR run')
