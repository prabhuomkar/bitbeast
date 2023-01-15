"""EfficientNet-B0 PyTorch Model Conversion"""
import sys

import torch
import torch_tensorrt


def save_torchscript(model, input):
    ts_model = torch.jit.trace(model, torch.randn(input).to(torch.device("cuda")).half())
    torch.jit.save(ts_model, "efficientnet_b0_ts_model.pt")

def save_trt_torchscript(model, input):
    trt_model = torch_tensorrt.compile(model,
        inputs=[torch_tensorrt.Input(input)],
        enabled_precisions={torch.half}
    )
    torch.jit.save(trt_model, "efficientnet_b0_trt_model.pt")

if __name__ == '__main__':
    print(f'torch: {torch.__version__}')
    print(f'torch-tensorrt: {torch_tensorrt.__version__}')
    print(f'cuda: {torch.cuda.is_available()}')
    model = torch.hub.load("pytorch/vision", "efficientnet_b0", weights="IMAGENET1K_V1")
    input = (1, 3, 224, 224)
    save_torchscript(model, input)
    save_trt_torchscript(model, input)
