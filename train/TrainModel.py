import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm


def train_MT_DCNN(*, model, data, label, lr=0.0001, epoch=2):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
    criterion = nn.MSELoss()
    LossRecord = []
    data = data.data
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, label)

        loss.backward()
        LossRecord.append(loss.item())
        optimizer.step()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    return model
