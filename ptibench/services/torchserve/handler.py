import numpy as np
import torch
import torch.nn.functional as F
from ts.torch_handler.image_classifier import ImageClassifier as PImageClassifier


class ImageClassifier(PImageClassifier):
    def preprocess(self, data):
        input = np.frombuffer(data[0]['data'], dtype=np.float32)
        shape = np.frombuffer(data[0]['shape'], dtype=int)
        input = torch.reshape(torch.from_numpy(input), tuple(shape))
        return input.unsqueeze(0).to(self.device)

    def postprocess(self, data):
        ps = F.softmax(data, dim=1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        return [[probs.tolist() + classes.tolist()]]