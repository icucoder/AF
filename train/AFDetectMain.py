import sys
sys.path.append("/root/AF")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

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
        self.classification = Classification()

    def forward(self, ecg, bcg): # shape:(N,1,length)
        ecg_feature = self.ecg_encoder(ecg)
        bcg_feature = self.bcg_encoder(bcg)
        ecg_ans = self.classification(ecg_feature)
        bcg_ans = self.classification(bcg_feature)
        ecg_restruct = self.ecg_decoder(ecg_feature)
        bcg_restruct = self.bcg_decoder(bcg_feature)
        # ecg_feature.shape:
        return ecg_feature, bcg_feature, ecg_ans, bcg_ans, ecg_restruct, bcg_restruct


def train_Encoder(*, model, ecg, bcg, label, lr=0.0001, epoch=2):
    criterion = nn.MSELoss()
    # crossloss = nn.CrossEntropyLoss()
    crossloss = nn.BCELoss()
    LossRecord = []
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()

        ecg_feature, bcg_feature, ecg_ans, bcg_ans, ecg_restruct, bcg_restruct = model(ecg, bcg)
        # loss1 = criterion(ecg_feature, bcg_feature)
        loss1 = criterion(ecg_feature[1:]-ecg_feature[:-1], bcg_feature[1:]-bcg_feature[:-1]) # 修改为BCG、ECG内部两两之间向量方向上的对齐
        loss2 = criterion(ecg_restruct, ecg) # 重构
        loss3 = criterion(bcg_restruct, bcg) # 重构
        loss4 = crossloss(ecg_ans.squeeze(1), label)  # AF检测
        # loss5 = crossloss(bcg_ans.squeeze(1), label)  # AF检测
        # print(loss1, loss2, loss3, loss4)
        # loss = loss1 + loss2 + loss3 + loss4# + loss5
        loss = loss1 + loss2 + loss3
        print(loss1, loss2, loss3)

        loss.backward()
        LossRecord.append(loss.item())
        optimizer.step()
        scheduler.step()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    plt.plot(LossRecord)
    plt.show()
    return model.cpu()


def get_DataSet():
    ECGPathList = [
        '004.chengjinqing.20180319.171534.37.ecg.af.csv',
        '007.songjinming.20180320.174932.37.ecg.af.csv',
        '009.caidabao.20180321.180258.35.ecg.af.csv',
        '012.zhuyunlong.20180321.185039.38.ecg.af.csv',
        '027.wuxiangguan.20180326.175519.35.ecg.af.csv',

        '001.wangqixiang.20180319.170246.38.ecg.af.csv',
        '002.zhufurong.20180319.164612.36.ecg.af.csv',
        '003.chenjinkang.20180319.170514.35.ecg.af.csv',
        '005.majialin.20180320.173223.35.ecg.af.csv',
        '006.wangxiangxinbao.20180320.174505.36.ecg.af.csv',
        '014.geyingdi.20180322.172818.37.ecg.af.csv',
        '016.lijinliang.20180323.164358.36.ecg.na.csv',
        '017.liaoyinghua.20180323.162433.37.ecg.na.csv',
        '018.wangruihua.20180323.164452.35.ecg.na.csv',
        '020.shenwenbao.20180324.174851.35.ecg.na.csv',
    ]
    BCGPathList = [
        '004.chengjinqing.20180319.171534.37.bcg.af.csv',
        '007.songjinming.20180320.174932.37.bcg.af.csv',
        '009.caidabao.20180321.180258.35.bcg.af.csv',
        '012.zhuyunlong.20180321.185039.38.bcg.af.csv',
        '027.wuxiangguan.20180326.175519.35.bcg.af.csv',

        '001.wangqixiang.20180319.170246.38.bcg.af.csv',
        '002.zhufurong.20180319.164612.36.bcg.af.csv',
        '003.chenjinkang.20180319.170514.35.bcg.af.csv',
        '005.majialin.20180320.173223.35.bcg.af.csv',
        '006.wangxiangxinbao.20180320.174505.36.bcg.af.csv',
        '014.geyingdi.20180322.172818.37.bcg.af.csv',
        '016.lijinliang.20180323.164358.36.bcg.na.csv',
        '017.liaoyinghua.20180323.162433.37.bcg.na.csv',
        '018.wangruihua.20180323.164452.35.bcg.na.csv',
        '020.shenwenbao.20180324.174851.35.bcg.na.csv',
    ]

    af_list = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(len(ECGPathList)):
        ECGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/ECG_cut/' + ECGPathList[i]
    for i in range(len(BCGPathList)):
        BCGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/BCG/' + BCGPathList[i]

    begin = 1000
    read_length = 10240
    slidingWindowSize = 2048
    ECG_vector = torch.zeros(0, 1, slidingWindowSize)
    BCG_vector = torch.zeros(0, 1, slidingWindowSize)
    for i in range(len(ECGPathList)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ECGPathList[i], begin=begin, length=read_length, column=2, isKansas=False)  # 标注采样hz 按时间读数据
        ECG_tmp = DataUtils.get_sliding_window_not_overlap(ECG, slidingWindowSize=slidingWindowSize).unsqueeze(1)
        ECG_vector = torch.cat([ECG_vector, ECG_tmp], dim=0)

        BCG = DataUtils.read_torch_from_CSV_data(path=BCGPathList[i], begin=begin, length=int(read_length * 1), column=2, isKansas=False)
        BCG = DataUtils.butter_bandpass_filter(BCG, 125, 1.0, 8.4)
        BCG_tmp = DataUtils.get_sliding_window_not_overlap(BCG, slidingWindowSize=slidingWindowSize).unsqueeze(1)
        BCG_vector = torch.cat([BCG_vector, BCG_tmp], dim=0)

    label = torch.zeros(ECG_vector.shape[0], 2, dtype=torch.float)
    onepersonnums = ECG_vector.shape[0] // len(ECGPathList)
    for i in range(len(af_list)):
        if (af_list[i] == 1):
            label[i * onepersonnums:(i + 1) * onepersonnums, 1] = 1  # [0, 1]  AF
        else:
            label[i * onepersonnums:(i + 1) * onepersonnums, 0] = 1  # [1, 0]  NAF
    ECG_vector = DataUtils.layernorm(ECG_vector)
    BCG_vector = DataUtils.layernorm(BCG_vector)
    return ECG_vector, BCG_vector, len(ECGPathList), label


def run_Encoder():
    ECG_vector, BCG_vector, persons, label = get_DataSet()

    # PltUtils.plot_all_data(ECG_vector)

    model = MyNet()
    model = train_Encoder(model=model.cuda(), ecg=ECG_vector.data.cuda(), bcg=BCG_vector.data.cuda(), label=label.cuda(), lr=0.0003, epoch=1000)

    ecg_feature, bcg_feature, ecg_ans, bcg_ans, ecg_restruct, bcg_restruct = model(ECG_vector, BCG_vector)
    print("ECG分类结果", ecg_ans)
    print("BCG分类结果", bcg_ans)

    torch.save(model, "../model/ResNetModel.pth")
    # ecg_feature = DataUtils.get_PCA_feature(ecg_feature.squeeze(1).detach().numpy(), 3)
    # bcg_feature = DataUtils.get_PCA_feature(bcg_feature.squeeze(1).detach().numpy(), 3)
    #
    # PltUtils.plot_3D_PCA_Figure(
    #     [
    #         ecg_feature[:ecg_feature.shape[0] // persons],
    #         ecg_feature[ecg_feature.shape[0] // persons:2 * ecg_feature.shape[0] // persons],
    #         ecg_feature[2 * ecg_feature.shape[0] // persons:3 * ecg_feature.shape[0] // persons],
    #         bcg_feature[:bcg_feature.shape[0] // persons],
    #         bcg_feature[bcg_feature.shape[0] // persons:2 * bcg_feature.shape[0] // persons],
    #         bcg_feature[2 * bcg_feature.shape[0] // persons:3 * bcg_feature.shape[0] // persons],
    #     ]
    # )

if __name__ == '__main__':
    run_Encoder()
