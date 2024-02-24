import struct
import numpy as np
import torch


model_state_dict = torch.load("assets/model.pth", map_location=torch.device("cpu"))
output_stream = open("assets/model.bin", "wb")

for name in model_state_dict.keys():
    data = model_state_dict[name].squeeze().numpy()
    print(f"{name}: {data}")
    # number of dimensions
    num_dims = len(data.shape)
    print(num_dims, data.shape)
    output_stream.write(struct.pack("i", num_dims))
    # dimension length and data
    data = data.astype(np.float32)
    for i in range(num_dims):
        output_stream.write(struct.pack("i", data.shape[num_dims - 1 - i]))
    data.tofile(output_stream)

output_stream.close()
