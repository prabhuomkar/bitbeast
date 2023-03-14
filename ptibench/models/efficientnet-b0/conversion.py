"""EfficientNet-B0 PyTorch Model Conversion"""
import sys

import torch
import torch_tensorrt


name = "efficientnet_b0"

def save_torchscript(model, input, is_half):
    input = torch.randn(input)
    input = input.to(torch.device("cuda"))
    if is_half:
        input = input.half()
    ts_model = torch.jit.trace(model, input)
    torch.jit.save(ts_model, f"ts_model_{'fp16' if is_half else 'fp32'}.pt")

def save_trt_torchscript(model, input, is_half):
    trt_model = torch_tensorrt.compile(model,
        inputs=[torch_tensorrt.Input(input, dtype=torch.half if is_half else torch.float32)],
        enabled_precisions={torch.half if is_half else torch.float32}
    )
    torch.jit.save(trt_model, f"model_{'fp16' if is_half else 'fp32'}.pt")

if __name__ == '__main__':
    print(f'torch: {torch.__version__}')
    print(f'torch-tensorrt: {torch_tensorrt.__version__}')
    print(f'cuda: {torch.cuda.is_available()}')
    model = torch.hub.load("pytorch/vision", name, weights="IMAGENET1K_V1").eval().to("cuda")
    is_half = False
    if len(sys.argv) > 1 and sys.argv[1] == 'half':
        is_half = True
        model = model.half().eval()
    print(f'half: {is_half}')
    input = (1, 3, 224, 224)
    save_trt_torchscript(model, input, is_half)
    save_torchscript(model, input, is_half)
