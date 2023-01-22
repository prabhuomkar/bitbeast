"""EfficientNet-B0 PyTorch Model Conversion"""
import os

import torch
import torch_tensorrt


name = "efficientnet_b0"

def save_torchscript(model, input):
    ts_model = torch.jit.trace(model, torch.randn(input).to(torch.device("cuda")))
    torch.jit.save(ts_model, f"ts_model.pt")

def save_trt_torchscript(model, input):
    trt_model = torch_tensorrt.compile(model,
        inputs=[torch_tensorrt.Input(input, dtype=torch.float)],
        enabled_precisions={torch.float}
    )
    torch.jit.save(trt_model, f"model.pt")

if __name__ == '__main__':
    print(f'torch: {torch.__version__}')
    print(f'torch-tensorrt: {torch_tensorrt.__version__}')
    print(f'cuda: {torch.cuda.is_available()}')
    model = torch.hub.load("pytorch/vision", name, weights="IMAGENET1K_V1").eval().to("cuda")
    input = (1, 3, 224, 224)
    save_trt_torchscript(model, input)
    save_torchscript(model, input)
