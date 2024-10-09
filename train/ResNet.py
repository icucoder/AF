import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils import DataUtils, PltUtils

torch.manual_seed(10)


class ECG_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 121
        # 编码
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

    def forward(self, data):
        output = self.conv1(data) + data
        output = DataUtils.layernorm(output)
        output = self.maxpool(output)
        output = self.conv2(output) + output
        output = self.maxpool(output)
        output = DataUtils.layernorm(output)
        output = self.conv3(output) + output
        output = DataUtils.layernorm(output)
        output = self.maxpool(output)
        return output


class BCG_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 121
        # 编码
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.maxpool = nn.MaxPool1d(kernel_size=2)

    def forward(self, data):
        output = self.conv1(data) + data
        output = DataUtils.layernorm(output)
        output = self.maxpool(output)
        output = self.conv2(output) + output
        output = self.maxpool(output)
        output = DataUtils.layernorm(output)
        output = self.conv3(output) + output
        output = DataUtils.layernorm(output)
        output = self.maxpool(output)
        return output


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.unconv1 = nn.ConvTranspose1d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.unconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.unconv3 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.unconv4 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        output = self.unconv1(input)
        output = DataUtils.layernorm(output)
        output = self.unconv2(output)
        output = DataUtils.layernorm(output)
        output = self.unconv3(output)
        output = DataUtils.layernorm(output)
        output = self.unconv4(output)
        output = DataUtils.layernorm(output)
        return output


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg_encoder = ECG_Encoder()
        self.bcg_encoder = BCG_Encoder()
        self.ecg_decoder = Decoder()
        self.bcg_decoder = Decoder()

    def forward(self, ecg, bcg):
        ecg_feature = self.ecg_encoder(ecg)
        bcg_feature = self.bcg_encoder(bcg)
        ecg = self.ecg_decoder(ecg_feature)
        bcg = self.bcg_decoder(bcg_feature)
        return ecg_feature, bcg_feature, ecg, bcg


def train_Encoder(*, model, ecg, bcg, lr=0.0001, epoch=2):
    criterion = nn.MSELoss()
    LossRecord = []
    for ep in tqdm(range(epoch)):
        if (ep % 100 == 0):
            lr = lr * 0.5
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
        optimizer.zero_grad()

        ecg_feature, bcg_feature, ecg1, bcg1 = model(ecg, bcg)
        loss1 = criterion(ecg_feature, bcg_feature)
        loss2 = criterion(ecg1, ecg)
        loss3 = criterion(bcg1, bcg)
        print(loss1, loss2, loss3)
        loss = loss1 + loss2 + loss3

        loss.backward()
        LossRecord.append(loss.item())
        optimizer.step()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    plt.plot(LossRecord)
    plt.show()
    return model


def run_Encoder():
    ECG_vector, BCG_vector, persons = DataUtils.get_DataSet()
    ECG_vector = DataUtils.layernorm(ECG_vector)
    BCG_vector = DataUtils.layernorm(BCG_vector)

    model = MyNet()
    model = train_Encoder(model=model, ecg=ECG_vector.data, bcg=BCG_vector.data, lr=0.1, epoch=2000)

    ecg_ans, bcg_ans, _, _ = model(ECG_vector, BCG_vector)
    print(ecg_ans.shape, bcg_ans.shape)

    ecg_feature = DataUtils.get_PCA_feature(ecg_ans.squeeze(1).detach().numpy(), 3)
    bcg_feature = DataUtils.get_PCA_feature(bcg_ans.squeeze(1).detach().numpy(), 3)

    PltUtils.plot_3D_PCA_Figure(
        [
            ecg_feature[:ecg_feature.shape[0] // persons],
            ecg_feature[ecg_feature.shape[0] // persons:2 * ecg_feature.shape[0] // persons],
            ecg_feature[2 * ecg_feature.shape[0] // persons:3 * ecg_feature.shape[0] // persons],
            bcg_feature[:bcg_feature.shape[0] // persons],
            bcg_feature[bcg_feature.shape[0] // persons:2 * bcg_feature.shape[0] // persons],
            bcg_feature[2 * bcg_feature.shape[0] // persons:3 * bcg_feature.shape[0] // persons],
        ]
    )


if __name__ == '__main__':
    run_Encoder()
