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


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # self.d_k = 1024
        self.linear1 = nn.Linear(256, 10000)
        # self.linear2 = nn.Linear(256, 10000)
        # self.linear3 = nn.Linear(256, 10000)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Sequential(
            # nn.Linear(10000, 10000),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(1),
            nn.Linear(10000, 256)
        )

    def forward(self, input):
        output = self.linear1(input)
        # k = self.linear2(input)
        # v = self.linear3(input)
        # output = torch.matmul(self.softmax(torch.matmul(output, k.transpose(-1, -2)) * self.d_k), v)
        output = self.fc(output)
        return output

# 用于ResNet 50/101/152的中间层block，其在内部变化时通道数目会发生改变（最终输出的通道数目时out_channels的四倍）
class BigBasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # ResNet 50/101/152的基本residual_block有三个卷积组成
        self.residual_block = nn.Sequential(
            # 第一个卷积用来调整输入通道数in_channels至out_channels
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BigBasicBlock.expansion, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential()
        # 如果维度或者通道数发生了改变，那这里shortcut路径需要使用卷积调整
        if stride != 1 or in_channels != out_channels * BigBasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BigBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BigBasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_block(x) + self.shortcut(x))

# 主体框架
class ResNet(nn.Module):
    def __init__(self, in_channels, block, num_block, num_classes):
        super().__init__()
        # 原文layer1这里使用的是大小为7的卷积，卷积核过大，效果不好，因此这里修改成大小为3卷积
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        # 默认进入到conv2_x的卷积个数都为64,所以self.current_channels为64
        self.current_channels = 64
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(block, out_channels=64, num_blocks=num_block[0], stride=1)
        self.conv3 = self._make_layer(block, out_channels=128, num_blocks=num_block[1], stride=2)
        self.conv4 = self._make_layer(block, out_channels=256, num_blocks=num_block[2], stride=2)
        self.conv5 = self._make_layer(block, out_channels=512, num_blocks=num_block[3], stride=2)
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    # 这里是用来产生中间层的block，其中block表示基本组成单元，由若干个（num_blocks个）smallBasicBlock或者bigBasicBlock组成
    # out_channels表示第一次需要调整的通道个数（最终basicBlock输出的通道数为这个的4倍），stride表示步长
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.current_channels, out_channels, stride))
            self.current_channels = out_channels * block.expansion # 1
        return nn.Sequential(*layers)


    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(self.maxPool(f1))
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(f4)
        output = self.softmax(self.fc(self.avgPool(f5).transpose(-1, -2)))
        return output

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upSampleSize=2, upSampleStride=2):
        super().__init__()

        self.residual_block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=upSampleSize,stride=upSampleStride),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

        self.shortcut = nn.Sequential()
        # 如果维度或者通道数发生了改变，那这里shortcut路径需要使用卷积调整
        if upSampleSize != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=upSampleSize, stride=upSampleStride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_block(x) + self.shortcut(x))

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


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg_encoder = ResNet(in_channels = 1, block = BigBasicBlock, num_block = [3, 4, 6, 3], num_classes = 256)
        self.bcg_encoder = ResNet(in_channels = 1, block = BigBasicBlock, num_block = [3, 4, 6, 3], num_classes = 256)
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
    data_loader1 = DataLoader(dataset=dataset1, batch_size=8, shuffle=True)
    data_loader2 = DataLoader(dataset=dataset2, batch_size=8, shuffle=True)
    for _ in tqdm(range(epoch)):

        for __, af_data_sample in enumerate(data_loader1, 1):
            optimizer.zero_grad()
            loss1, loss2, loss3, loss4, loss5, loss6 = 0, 0, 0, 0, 0, 0
            for __, naf_data_sample in enumerate(data_loader2, 1):
                # 16 1 2048  16 1 2048
                ecg_af_sample, bcg_af_sample = af_data_sample
                ecg_naf_sample, bcg_naf_sample = naf_data_sample
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

            loss1 *= 1.0
            loss2 *= 1.0
            loss3 *= 1.0
            loss4 *= 1.0
            loss5 *= 1.0
            loss6 *= 1.0
            print(loss1, loss2, loss3, loss4, loss5, loss6)
            loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

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
        ecg_af=ECG_AF_vector.data.cuda(),
        ecg_naf=ECG_NAF_vector.data.cuda(),
        bcg_af=BCG_AF_vector.data.cuda(),
        bcg_naf=BCG_NAF_vector.data.cuda(),
        lr=0.003,
        epoch=3000
    )

    # ecg_feature, bcg_feature, ecg_ans, bcg_ans, ecg_restruct, bcg_restruct = model(ECG_vector, BCG_vector)

    torch.save(model, "../model/ResNetModel.pth")
    print("训练结束，模型保存完成！")


if __name__ == '__main__':
    run_Encoder()
    # data = torch.rand(5,10,2048)
    # model = MyNet()
    # ecg_af_feature, bcg_af_feature, ecg_af_mlp, bcg_af_mlp, ecg_af_restruct, bcg_af_restruct = model(data, data)
    # print(ecg_af_feature.shape, bcg_af_feature.shape, ecg_af_mlp.shape, bcg_af_mlp.shape, ecg_af_restruct.shape, bcg_af_restruct.shape)

# 自对齐：把每个人内部的NAF/AF特征排列成线性
# 互对齐：将所有NAF聚集、将所有AF聚集
