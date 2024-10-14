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

data_root_path = 'H:/iScience/房颤数据/杭州原始数据/'

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


def train_Encoder(*, model, ecg_af, ecg_naf, bcg_af, bcg_naf, label, lr=0.001, epoch=2):
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
        optimizer.zero_grad()

        loss1 = 0
        loss2 = 0
        loss3 = 0
        loss4 = 0
        loss5 = 0
        loss6 = 0
        for __, af_data_sample in enumerate(data_loader1, 1):
            for __, naf_data_sample in enumerate(data_loader2, 1):
                # 16 1 2048  16 1 2048  16 2
                ecg_af_sample, bcg_af_sample = af_data_sample
                ecg_naf_sample, bcg_naf_sample = naf_data_sample
                # 16 1 256   16 1 256   16 1 256   16 1 256   16 1 2048
                ecg_af_feature, bcg_af_feature, ecg_af_mlp, bcg_af_mlp, ecg_af_restruct, bcg_af_restruct = model(ecg_af_sample, bcg_af_sample)
                print(ecg_af_feature.shape, bcg_af_feature.shape, ecg_af_mlp.shape, bcg_af_mlp.shape, ecg_af_restruct.shape, bcg_af_restruct.shape)
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


def get_AF_DataSet(begin, read_length, slidingWindowSize):
    ECGPathList = [
        '004.chengjinqing.20180319.171534.37.ecg.af.csv',
        '007.songjinming.20180320.174932.37.ecg.af.csv',
        '009.caidabao.20180321.180258.35.ecg.af.csv',
        '012.zhuyunlong.20180321.185039.38.ecg.af.csv',
        '027.wuxiangguan.20180326.175519.35.ecg.af.csv',
        '037.zhoudabao.20180412.175242.35.af.ecg.csv',
        '040.shenlaiying.20180412.184414.38.af.ecg.csv',
        '043.zhangxiangzhen.20180413.184228.38.af.ecg.csv',
        '047.zhengmeiying.20180416.193001.35.af.ecg.csv',
        '083.pinalin.20180612.204348.35.af.ecg.csv',
        '091.wanqibao.20180614.205249.35.af.ecg.csv',
    ]
    BCGPathList = [
        '004.chengjinqing.20180319.171534.37.bcg.af.csv',
        '007.songjinming.20180320.174932.37.bcg.af.csv',
        '009.caidabao.20180321.180258.35.bcg.af.csv',
        '012.zhuyunlong.20180321.185039.38.bcg.af.csv',
        '027.wuxiangguan.20180326.175519.35.bcg.af.csv',
        '037.zhoudabao.20180412.175242.35.af.bcg.csv',
        '040.shenlaiying.20180412.184414.38.af.bcg.csv',
        '043.zhangxiangzhen.20180413.184228.38.af.bcg.csv',
        '047.zhengmeiying.20180416.193001.35.af.bcg.csv',
        '083.pinalin.20180612.204348.35.af.bcg.csv',
        '091.wanqibao.20180614.205249.35.af.bcg.csv',
    ]

    for i in range(len(ECGPathList)):
        ECGPathList[i] = data_root_path + 'ECG_cut/' + ECGPathList[i]
    for i in range(len(BCGPathList)):
        BCGPathList[i] = data_root_path + 'BCG/' + BCGPathList[i]

    # begin = 1000
    # read_length = 10240
    # slidingWindowSize = 1024
    ECG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    BCG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    for i in range(len(ECGPathList)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ECGPathList[i], begin=begin, length=read_length, f=200, column=2)  # 标注采样hz 按时间读数据
        ECG_tmp = DataUtils.get_sliding_window_not_overlap(ECG, slidingWindowSize=slidingWindowSize).unsqueeze(0)
        ECG_vector = torch.cat([ECG_vector, ECG_tmp], dim=0)

        BCG = DataUtils.read_torch_from_CSV_data(path=BCGPathList[i], begin=begin, length=read_length, f=125, column=2)
        BCG = DataUtils.butter_bandpass_filter(BCG, 125, 1.0, 8.4)
        BCG_tmp = DataUtils.get_sliding_window_not_overlap(BCG, slidingWindowSize=slidingWindowSize).unsqueeze(0)
        BCG_vector = torch.cat([BCG_vector, BCG_tmp], dim=0)

    label = torch.zeros(ECG_vector.shape[0], 2, dtype=torch.float)
    onepersonnums = ECG_vector.shape[0] // len(ECGPathList)
    for i in range(len(ECGPathList)):
        label[i * onepersonnums:(i + 1) * onepersonnums, 1] = 1  # [0, 1]  AF
    ECG_vector = DataUtils.layernorm(ECG_vector)
    BCG_vector = DataUtils.layernorm(BCG_vector)
    return ECG_vector, BCG_vector, len(ECGPathList), label


def get_NAF_DataSet(begin, read_length, slidingWindowSize):
    ECGPathList = [
        '008.linlaiying.20180320.175323.38.ecg.na.csv',
        '013.yushuizhen.20180322.172202.36.ecg.na.csv',
        '016.lijinliang.20180323.164358.36.ecg.na.csv',
        '017.liaoyinghua.20180323.162433.37.ecg.na.csv',
        '018.wangruihua.20180323.164452.35.ecg.na.csv',
        '020.shenwenbao.20180324.174851.35.ecg.na.csv',
        '021.sunjugen.20180324.181212.37.ecg.na.csv',
        '022.lincuiguan.20180324.180026.36.ecg.na.csv',
        '023.wangzhaofa.20180325.175901.35.ecg.na.csv',
        '024.chengjinfang.20180325.182828.37.ecg.na.csv',
        '025.chenrenxing.20180325.182125.36.ecg.na.csv',
        '026.shenying.20180326.181246.36.ecg.na.csv',
        '028.luamei.20180326.182402.37.ecg.na.csv',
        '029.shichenhao.20180327.233252.36.ecg.na.csv',
        '030.zhanghaiqiang.20180328.224655.36.ecg.na.csv',
        '031.yubin.20180329.191337.36.ecg.na.csv',
        '045.chensuhua.20180414.180932.35.na.ecg.csv',
        '046.wujinhua.20180414.185039.37.na.ecg.csv',
        '049.xiafurong.20180416.200429.37.na.ecg.csv',
        '053.yaoazhao.20180417.185423.37.na.ecg.csv',
        '054.xufurong.20180417.190646.38.na.ecg.csv',
        '056.guyafu.20180418.191454.36.na.ecg.csv',
        '057.wuhongde.20180418.185107.37.na.ecg.csv',
        '059.taoshouting.20180419.185644.35.na.ecg.csv',
        '065.yusanjian.20180420.193147.37.na.ecg.csv',
        '069.geyongzhi.20180422.195719.37.na.ecg.csv',
        '070.wuchuanyong.20180422.200924.38.na.ecg.csv',
        '072.xuliugen.20180423.193038.36.na.ecg.csv',
        '075.panqijin.20180424.193717.35.na.ecg.csv',
        '077.wujinyu.20180424.195153.37.na.ecg.csv',
        '078.yushuigen.20180424.192604.38.na.ecg.csv',
    ]
    BCGPathList = [
        '008.linlaiying.20180320.175323.38.bcg.na.csv',
        '013.yushuizhen.20180322.172202.36.bcg.na.csv',
        '016.lijinliang.20180323.164358.36.bcg.na.csv',
        '017.liaoyinghua.20180323.162433.37.bcg.na.csv',
        '018.wangruihua.20180323.164452.35.bcg.na.csv',
        '020.shenwenbao.20180324.174851.35.bcg.na.csv',
        '021.sunjugen.20180324.181212.37.bcg.na.csv',
        '022.lincuiguan.20180324.180026.36.bcg.na.csv',
        '023.wangzhaofa.20180325.175901.35.bcg.na.csv',
        '024.chengjinfang.20180325.182828.37.bcg.na.csv',
        '025.chenrenxing.20180325.182125.36.bcg.na.csv',
        '026.shenying.20180326.181246.36.bcg.na.csv',
        '028.luamei.20180326.182402.37.bcg.na.csv',
        '029.shichenhao.20180327.233252.36.bcg.na.csv',
        '030.zhanghaiqiang.20180328.224655.36.bcg.na.csv',
        '031.yubin.20180329.191337.36.bcg.na.csv',
        '045.chensuhua.20180414.180932.35.na.bcg.csv',
        '046.wujinhua.20180414.185039.37.na.bcg.csv',
        '049.xiafurong.20180416.200429.37.na.bcg.csv',
        '053.yaoazhao.20180417.185423.37.na.bcg.csv',
        '054.xufurong.20180417.190646.38.na.bcg.csv',
        '056.guyafu.20180418.191454.36.na.bcg.csv',
        '057.wuhongde.20180418.185107.37.na.bcg.csv',
        '059.taoshouting.20180419.185644.35.na.bcg.csv',
        '065.yusanjian.20180420.193147.37.na.bcg.csv',
        '069.geyongzhi.20180422.195719.37.na.bcg.csv',
        '070.wuchuanyong.20180422.200924.38.na.bcg.csv',
        '072.xuliugen.20180423.193038.36.na.bcg.csv',
        '075.panqijin.20180424.193717.35.na.bcg.csv',
        '077.wujinyu.20180424.195153.37.na.bcg.csv',
        '078.yushuigen.20180424.192604.38.na.bcg.csv',
    ]

    for i in range(len(ECGPathList)):
        ECGPathList[i] = data_root_path + 'ECG_cut/' + ECGPathList[i]
    for i in range(len(BCGPathList)):
        BCGPathList[i] = data_root_path + 'BCG/' + BCGPathList[i]

    # begin = 1000
    # read_length = 10240
    # slidingWindowSize = 2048
    ECG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    BCG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    for i in range(len(ECGPathList)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ECGPathList[i], begin=begin, length=read_length, f=200, column=2)  # 标注采样hz 按时间读数据
        ECG_tmp = DataUtils.get_sliding_window_not_overlap(ECG, slidingWindowSize=slidingWindowSize).unsqueeze(0)
        ECG_vector = torch.cat([ECG_vector, ECG_tmp], dim=0)

        BCG = DataUtils.read_torch_from_CSV_data(path=BCGPathList[i], begin=begin, length=read_length, f=125, column=2)
        BCG = DataUtils.butter_bandpass_filter(BCG, 125, 1.0, 8.4)
        BCG_tmp = DataUtils.get_sliding_window_not_overlap(BCG, slidingWindowSize=slidingWindowSize).unsqueeze(0)
        BCG_vector = torch.cat([BCG_vector, BCG_tmp], dim=0)

    label = torch.zeros(ECG_vector.shape[0], 2, dtype=torch.float)
    onepersonnums = ECG_vector.shape[0] // len(ECGPathList)
    for i in range(len(ECGPathList)):
        label[i * onepersonnums:(i + 1) * onepersonnums, 0] = 1  # [1, 0]  NAF
    ECG_vector = DataUtils.layernorm(ECG_vector)
    BCG_vector = DataUtils.layernorm(BCG_vector)
    return ECG_vector, BCG_vector, len(ECGPathList), label


def run_Encoder():
    begin = 1000
    read_length = 10240
    slidingWindowSize = 2048
    ECG_AF_vector, BCG_AF_vector, AF_persons, AF_label = get_AF_DataSet(begin, read_length, slidingWindowSize)
    ECG_NAF_vector, BCG_NAF_vector, NAF_persons, NAF_label = get_NAF_DataSet(begin, read_length, slidingWindowSize)

    ECG_vector = torch.cat([ECG_AF_vector, ECG_NAF_vector], dim=0)
    BCG_vector = torch.cat([BCG_AF_vector, BCG_NAF_vector], dim=0)
    label = torch.cat([AF_label, NAF_label], dim=0)

    print(ECG_vector.shape, BCG_vector.shape)
    # PltUtils.plot_all_data(ECG_vector)

    model = MyNet()
    model = train_Encoder(
        model=model.cuda(),
        ecg_af=ECG_AF_vector.data.cuda(),
        ecg_naf=ECG_NAF_vector.data.cuda(),
        bcg_af=BCG_AF_vector.data.cuda(),
        bcg_naf=BCG_NAF_vector.data.cuda(),
        label=label.cuda(),
        lr=0.0003,
        epoch=1000
    )

    # ecg_feature, bcg_feature, ecg_ans, bcg_ans, ecg_restruct, bcg_restruct = model(ECG_vector, BCG_vector)

    torch.save(model, "../model/ResNetModel.pth")
    print("训练结束，模型保存完成！")


if __name__ == '__main__':
    run_Encoder()

# 自对齐：把每个人内部的NAF/AF特征排列成线性
# 互对齐：将所有NAF聚集、将所有AF聚集
