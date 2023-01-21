# PTIBench: PyTorch Inference Server Benchmark

## About
This repository is a playground to benchmark several existing (and popular) model serving frameworks for PyTorch.  
It also acts as a reference kit to generate, run and benchmark your own model in available frameworks. 

## Models

| Model Name | Precision | GPU | Type |
| ---------- | --------- | --- | ---- |
| EfficientNet-B0 | FP32 | True | NVIDIA Triton Inference Server |
| EfficientNet-B0 | FP32 | True | PyTorch Serve |

### Generate and Usage
This section will help you compile and generate modules which can be served.

- Compile and generate assets to be used for serving using the command:
```
cd models/efficientnet-b0
docker run --rm -it --gpus all -v ${PWD}:/scratch_space nvcr.io/nvidia/pytorch:<xx.yy>-py3 # e.g. <xx.yy> = 22.05
cd /scratch_space
python3 conversion.py
exit
```
- Copy the triton model in model_repository:
```
mv model.pt ../services/triton/model_repository/efficientnet_b0/1/model.pt
cp ts_model.pt ../services/torchserve/ts_model.pt
```

#### NVIDIA Triton Inference Server
- Run NVIDIA Triton Inference Server using:
```
cd services/triton
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v <absolute_path_to_ptibench_directory>/model_repository:/models \
    nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models # e.g. <xx.yy> = 22.05
```

#### TorchServe
- Generate MAR for running TorchServe:
```
torch-model-archiver --model-name efficientnet_b0 --version 1.0 --serialized-file ts_model.pt --handler handler.py
```
- Run TorchServe using:
```
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 -v ${PWD}/model-store:/home/model-server/model-store \
    pytorch/torchserve:latest-gpu torchserve --model-store /home/model-server/model-store/ --models efficientnet_b0=efficientnet_b0.mar
```

### Running Benchmarks
This section will help you run benchmark and save results

TODO(omkar)

## TODOs/Improvements
- Run benchmarks weekly/on latest versions using common GPU spec
- Use FP16 for inference as well
- Generate some fancy graphs using the saved results 

## License
This project is licensed under [MIT License](LICENSE).