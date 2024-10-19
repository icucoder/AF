import torch.nn as nn
import torch
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

# 编码器的基本block，这里采用Resnet50/101/152的BigBasicBlock版本
# 三层卷积：第一层负责将in_channels->out_channels，第二层负责降采样，第三层负责将out_channels-> expansion * out_channels
# 这么做本质上是一个瓶颈结构，先扩大特征选择面，再通过第二层卷积滤除无效特征，最后通过第三层卷积被留下的特征
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.shortcut = nn.Sequential()
        # 如果维度或者通道数发生了改变，那这里shortcut路径需要使用卷积调整
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_block(x) + self.shortcut(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upSampleSize=2, upSampleStride=2):
        super().__init__()

        self.residual_block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=upSampleSize, stride=upSampleStride),
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


class ResUnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.Encoder1 = EncoderBlock(in_channels=1, out_channels=64)
        self.Encoder2 = EncoderBlock(in_channels=64, out_channels=128)
        self.Encoder3 = EncoderBlock(in_channels=128, out_channels=256)
        self.Encoder4 = EncoderBlock(in_channels=256, out_channels=512)
        self.BottleEncoder = EncoderBlock(in_channels=512, out_channels=1024)

        self.downSample = nn.MaxPool1d(kernel_size=2)

        self.upSampleBlock4 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.Decoder4 = EncoderBlock(in_channels=1024, out_channels=512)

        self.upSampleBlock3 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.Decoder3 = EncoderBlock(in_channels=512, out_channels=256)

        self.upSampleBlock2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.Decoder2 = EncoderBlock(in_channels=256, out_channels=128)

        self.upSampleBlock1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.Decoder1 = EncoderBlock(in_channels=128, out_channels=64)

        self.toOutput = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        down1_conn = self.Encoder1(x)
        down1 = self.downSample(down1_conn)

        down2_conn = self.Encoder2(down1)
        down2 = self.downSample(down2_conn)

        down3_conn = self.Encoder3(down2)
        down3 = self.downSample(down3_conn)

        down4_conn = self.Encoder4(down3)
        down4 = self.downSample(down4_conn)

        mid_out = self.BottleEncoder(down4)

        up5 = self.Decoder4(torch.cat([down4_conn, self.upSampleBlock4(mid_out)], dim=1))
        up4 = self.Decoder3(torch.cat([down3_conn, self.upSampleBlock3(up5)], dim=1))
        up3 = self.Decoder2(torch.cat([down2_conn, self.upSampleBlock2(up4)], dim=1))
        up2 = self.Decoder1(torch.cat([down1_conn, self.upSampleBlock1(up3)], dim=1))
        reConstruct = self.toOutput(up2)
        return reConstruct, down1_conn, down2_conn, down3_conn, down4_conn, mid_out


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(131072, 2048),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1),
            nn.Linear(2048, 1024)
        )

    def flatten_data(self, x):
        return x.flatten(start_dim=1)

    def forward(self, input):
        return self.fc(self.flatten_data(input).unsqueeze(1))


class ConcatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = MLP()
        self.mlp2 = MLP()
        self.mlp3 = MLP()
        self.mlp4 = MLP()
        self.mlp5 = MLP()

    # down1_conn.shape: N 64 length
    # down2_conn.shape: N 128 length/2
    # down3_conn.shape: N 256 length/4
    # down4_conn.shape: N 512 length/8
    #    mid_out.shape: N 1024 length/16
    # return feature.shape: N 1 length/16*31
    def forward(self, down1_conn, down2_conn, down3_conn, down4_conn, mid_out):
        down1_conn = self.mlp1(down1_conn)
        down2_conn = self.mlp2(down2_conn)
        down3_conn = self.mlp3(down3_conn)
        down4_conn = self.mlp4(down4_conn)
        mid_out = self.mlp5(mid_out)
        feature = down1_conn + down2_conn + down3_conn + down4_conn + mid_out
        return feature


class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resUnet = ResUnet()
        self.concatNet = ConcatNet()

    def forward(self, x):
        person, nums, length = x.shape
        x = x.reshape(person * nums, 1, length)
        reConstruct, down1_conn, down2_conn, down3_conn, down4_conn, mid_out = self.resUnet(x)
        feature = self.concatNet(down1_conn, down2_conn, down3_conn, down4_conn, mid_out)
        reConstruct = reConstruct.reshape(person, nums, length)
        feature = feature.reshape(person, nums, feature.shape[-1])
        return reConstruct, feature


class UnitedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = EncoderNet()
        self.encoder2 = EncoderNet()

    def forward(self, ecg, bcg):
        ecg_re, ecg_f = self.encoder1(ecg)
        bcg_re, bcg_f = self.encoder2(bcg)
        return ecg_re, bcg_re, ecg_f, bcg_f


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
                loss1, loss2, loss3, loss4, loss5, loss6, loss7 = 0, 0, 0, 0, 0, 0, 0
                # 16 1 2048  16 1 2048
                ecg_af_sample, bcg_af_sample = af_data_sample
                ecg_naf_sample, bcg_naf_sample = naf_data_sample
                ecg_af_sample = ecg_af_sample.cuda()
                bcg_af_sample = bcg_af_sample.cuda()
                ecg_naf_sample = ecg_naf_sample.cuda()
                bcg_naf_sample = bcg_naf_sample.cuda()
                # 16 1 256   16 1 256   16 1 256   16 1 256   16 1 2048
                ecg_af_restruct, bcg_af_restruct, ecg_af_mlp, bcg_af_mlp  = model(ecg_af_sample, bcg_af_sample)
                ecg_naf_restruct, bcg_naf_restruct, ecg_naf_mlp, bcg_naf_mlp  = model(ecg_naf_sample, bcg_naf_sample)
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
                # margin = 10
                # loss7 += DataUtils.MetricLoss(ecg_naf_mlp, ecg_af_mlp, margin) + DataUtils.MetricLoss(bcg_naf_mlp, bcg_af_mlp, margin)

                # 先重构提取本质特征  增强模型泛化性  引入非持续性房颤数据（观测是否为线性？）

                loss1 *= 10.0
                loss2 *= 1.0
                loss3 *= 100.0
                loss4 *= 0.0
                loss5 *= 0.01
                loss6 *= 0.01
                loss7 *= 10.0
                print(loss1, loss2, loss3, loss4, loss5, loss6, loss7)
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

                loss.backward()
                LossRecord.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    plt.plot(LossRecord)
    plt.show()
    return model.cpu()


if __name__ == '__main__':
    # data = torch.zeros(5, 10, 2048)
    ECG_AF_vector = torch.load("../dataset/ECG_AF_vector.pth")
    BCG_AF_vector = torch.load("../dataset/BCG_AF_vector.pth")
    ECG_NAF_vector = torch.load("../dataset/ECG_NAF_vector.pth")
    BCG_NAF_vector = torch.load("../dataset/BCG_NAF_vector.pth")
    model = UnitedNet()
    # ecg_re, ecg_f, bcg_re, bcg_f = model(data, data)
    model = train_Encoder(
        model=model.cuda(),
        ecg_af=ECG_AF_vector.data,
        ecg_naf=ECG_NAF_vector.data,
        bcg_af=BCG_AF_vector.data,
        bcg_naf=BCG_NAF_vector.data,
        lr=0.0003,
        epoch=3
    )
    torch.save(model, "../model/UnitedNetModel.pth")
    print("训练结束，模型保存完成！")