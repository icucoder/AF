import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import DataUtils, PltUtils

torch.manual_seed(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(out_channels),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=2),
            nn.BatchNorm1d(out_channels),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2)

    def forward(self, data):  # N in_channels f
        # 残差
        output = self.conv1(data)  # N in_channels f
        output = self.conv2(output)  # N out_channels f
        output = self.conv3(output)  # N out_channels f
        output = self.maxpool(output)  # N in_channels f/2
        # 保留信息
        data = self.conv4(data)  # N out_channels f/2
        return output + data


class ECG_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet1 = ResNet(1, 64, 11)
        self.resnet2 = ResNet(64, 128, 11)
        self.resnet3 = ResNet(128, 256, 11)
        self.onepool = nn.AdaptiveAvgPool1d(1)

    def forward(self, data):  # N 1 f
        output = self.resnet1(data)  # N 64 f/2
        output = self.resnet2(output)  # N 128 f/4
        output = self.resnet3(output)  # N 256 f/8
        output = self.onepool(output)  # N 256 1
        return output.transpose(-1, -2)  # N 1 256


class UpSampleResNet(nn.Module):
    def __init__(self, in_channels, out_channels, N):  # N表示上采样倍数
        super().__init__()
        if N == 1:
            self.unconv1 = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm1d(out_channels),
                # nn.ReLU()
            )
        else:
            self.unconv1 = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, 2 * N, stride=N, padding=N // 2),
                nn.BatchNorm1d(out_channels),
                # nn.ReLU()
            )
        self.unconv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, padding='same', padding_mode='reflect'),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, data):
        data = self.unconv1(data)
        data = self.unconv2(data)
        return data


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.unconv1 = UpSampleResNet(in_channels=1, out_channels=64, N=2)
        self.unconv2 = UpSampleResNet(in_channels=64, out_channels=32, N=2)
        self.unconv3 = UpSampleResNet(in_channels=32, out_channels=1, N=2)

    def forward(self, input):
        output = self.unconv1(input)
        output = self.unconv2(output)
        output = self.unconv3(output)
        output = DataUtils.layernorm(output)
        return output


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # self.d_k = 1024
        # self.linear2 = nn.Linear(256, 10000)
        # self.linear3 = nn.Linear(256, 10000)
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 10000),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1),
            nn.Linear(10000, 256)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            # nn.LeakyReLU(),
            nn.BatchNorm1d(1),
            nn.Linear(256, 64)
        )

    def forward(self, input):
        output = self.fc1(input)
        output = self.fc2(output)
        return output


class Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 10000)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(10000, 2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        output = self.linear1(input)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.softmax(output)
        return output


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg_encoder = ECG_Encoder()
        self.bcg_encoder = ECG_Encoder()
        self.ecg_decoder = Decoder()
        self.bcg_decoder = Decoder()
        # self.classification = Classification()
        self.mlp1 = MLP()
        self.mlp2 = MLP()

    def forward(self, ecg_origin, bcg_origin):  # shape:(N,num,length)
        ecg = ecg_origin.reshape(ecg_origin.shape[0] * ecg_origin.shape[1], 1, ecg_origin.shape[-1])
        bcg = bcg_origin.reshape(bcg_origin.shape[0] * bcg_origin.shape[1], 1, bcg_origin.shape[-1])
        ecg_feature = self.ecg_encoder(ecg)
        bcg_feature = self.bcg_encoder(bcg)
        ecg_mlp = self.mlp1(ecg_feature)
        bcg_mlp = self.mlp2(bcg_feature)
        ecg_restruct = self.ecg_decoder(ecg_feature)
        bcg_restruct = self.bcg_decoder(bcg_feature)
        return (
            ecg_feature.reshape(ecg_origin.shape[0], ecg_origin.shape[1], ecg_feature.shape[-1]),
            bcg_feature.reshape(bcg_origin.shape[0], bcg_origin.shape[1], bcg_feature.shape[-1]),
            ecg_mlp.reshape(ecg_origin.shape[0], ecg_origin.shape[1], ecg_mlp.shape[-1]),
            bcg_mlp.reshape(bcg_origin.shape[0], bcg_origin.shape[1], bcg_mlp.shape[-1]),
            ecg_restruct.reshape(ecg_origin.shape[0], ecg_origin.shape[1], ecg_restruct.shape[-1]),
            bcg_restruct.reshape(bcg_origin.shape[0], bcg_origin.shape[1], bcg_restruct.shape[-1])
        )


def train_Encoder(*, model, ecg_af, ecg_naf, bcg_af, bcg_naf, lr=0.001, epoch=2):
    criterion = nn.MSELoss()
    # crossloss = nn.CrossEntropyLoss()
    crossloss = nn.BCELoss()
    LossRecord = []
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
    dataset1 = TensorDataset(ecg_af, bcg_af)
    dataset2 = TensorDataset(ecg_naf, bcg_naf)
    data_loader1 = DataLoader(dataset=dataset1, batch_size=6, shuffle=True)
    data_loader2 = DataLoader(dataset=dataset2, batch_size=6, shuffle=True)
    for _ in tqdm(range(epoch)):
        for __, af_data_sample in enumerate(data_loader1, 1):
            for __, naf_data_sample in enumerate(data_loader2, 1):
                optimizer.zero_grad()
                loss1, loss2, loss3, loss4, loss5, loss6, loss7 = 0, 0, 0, 0, 0, 0, 0
                # 16 1 2048  16 1 2048
                ecg_af_sample, bcg_af_sample = af_data_sample
                ecg_naf_sample, bcg_naf_sample = naf_data_sample
                ecg_af_sample = ecg_af_sample.cuda()
                bcg_af_sample = bcg_af_sample.cuda()
                ecg_naf_sample = ecg_naf_sample.cuda()
                bcg_naf_sample = bcg_naf_sample.cuda()
                # 16 1 256   16 1 256   16 1 256   16 1 256   16 1 2048
                ecg_af_feature, bcg_af_feature, ecg_af_mlp, bcg_af_mlp, ecg_af_restruct, bcg_af_restruct = model(ecg_af_sample, bcg_af_sample)
                # print(ecg_af_feature.shape, bcg_af_feature.shape, ecg_af_mlp.shape, bcg_af_mlp.shape, ecg_af_restruct.shape, bcg_af_restruct.shape)
                ecg_naf_feature, bcg_naf_feature, ecg_naf_mlp, bcg_naf_mlp, ecg_naf_restruct, bcg_naf_restruct = model(ecg_naf_sample, bcg_naf_sample)
                # 自对齐（连续性）
                loss1 += DataUtils.continuity_loss([ecg_af_mlp, bcg_af_mlp, ecg_naf_mlp, bcg_naf_mlp])
                # 互对齐
                loss2 += DataUtils.CLIP_loss(ecg_naf_mlp, ecg_af_mlp) + DataUtils.CLIP_loss(bcg_naf_mlp, bcg_af_mlp)
                # BCG方向与ECG方向对齐
                loss3 += criterion(DataUtils.CLIP_metric(ecg_naf_mlp, ecg_af_mlp), DataUtils.CLIP_metric(bcg_naf_mlp, bcg_af_mlp))
                # 按时间对齐提取到的特征
                loss4 += criterion(ecg_af_mlp, bcg_af_mlp) + criterion(ecg_naf_mlp, bcg_naf_mlp)
                # 重构
                loss5 += criterion(ecg_af_restruct, ecg_af_sample) + criterion(ecg_naf_restruct, ecg_naf_sample)
                loss6 += criterion(bcg_af_restruct, bcg_af_sample) + criterion(bcg_naf_restruct, bcg_naf_sample)
                # 尝试添加三元组损失margin  绘制图中最好能够将每一个数据点对应的原片段绘制出来辅助观测是否真的为AF
                margin = 10
                loss7 += DataUtils.MetricLoss(ecg_naf_mlp, ecg_af_mlp, margin) + DataUtils.MetricLoss(bcg_naf_mlp, bcg_af_mlp, margin)

                loss1 *= 1.0
                loss2 *= 100.0
                loss3 *= 10.0
                loss4 *= 0.0
                loss5 *= 1.0
                loss6 *= 1.0
                loss7 *= 10.0
                print(loss1, loss2, loss3, loss4, loss5, loss6, loss7)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

                loss.backward()
                LossRecord.append(loss.item())
                optimizer.step()

        scheduler.step()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    plt.plot(LossRecord)
    plt.show()
    return model.cpu()





def run_Encoder():
    # begin = 1000
    # read_length = 10240
    # slidingWindowSize = 2048
    # ECG_AF_vector, BCG_AF_vector, AF_persons, AF_label = get_AF_DataSet(begin, read_length, slidingWindowSize)
    # ECG_NAF_vector, BCG_NAF_vector, NAF_persons, NAF_label = get_NAF_DataSet(begin, read_length, slidingWindowSize)

    ECG_AF_vector = torch.load("../dataset/ECG_AF_vector.pth")
    BCG_AF_vector = torch.load("../dataset/BCG_AF_vector.pth")
    ECG_NAF_vector = torch.load("../dataset/ECG_NAF_vector.pth")
    BCG_NAF_vector = torch.load("../dataset/BCG_NAF_vector.pth")

    ECG_vector = torch.cat([ECG_AF_vector, ECG_NAF_vector], dim=0)
    BCG_vector = torch.cat([BCG_AF_vector, BCG_NAF_vector], dim=0)

    print(ECG_vector.shape, BCG_vector.shape)

    model = MyNet()
    model = train_Encoder(
        model=model.cuda(),
        ecg_af=ECG_AF_vector.data,
        ecg_naf=ECG_NAF_vector.data,
        bcg_af=BCG_AF_vector.data,
        bcg_naf=BCG_NAF_vector.data,
        lr=0.0003,
        epoch=3000
    )

    # ecg_feature, bcg_feature, ecg_ans, bcg_ans, ecg_restruct, bcg_restruct = model(ECG_vector, BCG_vector)

    torch.save(model, "../model/ResNetModel.pth")
    print("训练结束，模型保存完成！")


if __name__ == '__main__':
    run_Encoder()

# 自对齐：把每个人内部的NAF/AF特征排列成线性
# 互对齐：将所有NAF聚集、将所有AF聚集
