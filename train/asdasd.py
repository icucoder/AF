import torch.nn as nn
import torch

data = torch.arange(25, dtype=torch.float).reshape(5,1,5)

N=2
model = nn.ConvTranspose1d(1, 4, 2 * N, stride=N, padding=N // 2)

print(model(data).shape)