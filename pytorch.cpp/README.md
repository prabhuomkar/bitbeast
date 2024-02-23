# pytorch.cpp
Running PyTorch Models for Inference using GGML

## Directory Structure
- [conversion.py](conversion.py) - Converts weights of a PyTorch model to GGML format
- [model.py](model.py) - Sample PyTorch model for training a neural network to learn 2 input truth table
- [main.cpp](main.cpp) - Main driver program for running inference using ggml

## Getting Started 

### Train Model
- Run the following command to train the model on your choice of truth table:
```
python3 model.py xor
OR
python3 model.py and
OR
python3 model.py or
```
- This will save the weights in the `assets` folder with name `model.pth`

### Convert PyTorch Model Weights to GGUF
- [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) is a binary format that is designed for fast loading and saving of models, and for ease of reading.
- Usually your model weights per layer are stored with dimensions shape, actual dimensions and then actual weights tightly.
- Packing needs to be done as per GGUF spec so that it can be loaded using GGML code.
- Run following command to convert your PyTorch model weights stored in `assets/model.pth` to GGUF:
```
python3 conversion.py
```

### Compile and Run Inference using GGML
- Refer [main.cpp](main.cpp) for referring to `load` and `predict` functions.
- `load` loads the GGUF and reads the weights to initialize GGML params per layer specific to the model and initialize context.
- `predict` uses the initialized model and perform the vector calculations as a forward pass would do eventually.
- Run following command to include GGML headers:
```
git clone https://github.com/ggerganov/ggml
```
- Compile and create the excutable for running the inference:
```
mkdir build && cd build
cmake ..
make
```
- Run the inference:
```
./bin/pytorch.cpp
```

## License
This project is licensed under [MIT License](LICENSE).