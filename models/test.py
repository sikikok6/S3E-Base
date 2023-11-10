from resnetrgb import resnet18

import torch

# data = {"images": torch.randn((1, 3, 600, 600))}
data = {"images": torch.randn((1, 3, 480, 640))}

model = resnet18()

model(data)
