from cProfile import label

import torch
import torch.nn as nn

from utils import DataUtils


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3)

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.maxpool2(data)
        data = self.conv3(data)
        data = self.maxpool2(data)
        data = self.conv4(data)
        data = self.maxpool2(data)
        data = self.conv5(data)
        data = self.maxpool2(data)
        data = self.conv6(data)
        data = self.maxpool3(data)
        return data


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.unconv1 = nn.ConvTranspose1d(in_channels=1, out_channels=128, kernel_size=12, stride=6, padding=3)
        self.unconv2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.unconv3 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.unconv4 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.unconv5 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        output = self.unconv1(input)
        output = self.unconv2(output)
        output = self.unconv3(output)
        output = self.unconv4(output)
        output = self.unconv5(output)
        return output


class Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(80, 32)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        output = self.linear1(input)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.softmax(output)
        return output


class MT_DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classify = Classification()

    def forward(self, input):
        output1 = self.encoder(input)
        print(output1.shape)
        classAns = self.classify(output1)
        print(classAns.shape)
        output2 = self.decoder(output1)
        print(output2.shape)
        return classAns

def get_label_from_ECG(ECG, slidingWindowSize):
    beatsWindows = 5
    label = torch.zeros(ECG.shape[0], 2)  # [0.3, 0.7]表示30%的概率正常、70%的概率房颤
    label[:,0] = 1 # 初始化
    R_index_list = DataUtils.get_R_index_from_ECG(ECG)
    hrv_list = DataUtils.get_RR_diff_from_R_index(R_index_list, windows=beatsWindows)
    print(hrv_list)
    AF_range_list = []
    for i in range(len(hrv_list)):
        if (hrv_list[i] >= 4):
            AF_range_list.append([R_index_list[i], R_index_list[i+beatsWindows]])
    return label

def run_MT_DCNN():
    length = 10000
    slidingWindowSize = 3840
    ECG = DataUtils.read_torch_from_CSV_data(path='H:/iScience/房颤数据/Kansas房颤/Original_ECG_X10036_350000.csv', begin=10, length=length, column=1)
    label = get_label_from_ECG(ECG, slidingWindowSize)

    # data = DataUtils.get_sliding_window(ECG, slidingWindowSize=3840)
    # train = MT_DCNN()
    return

if __name__ == '__main__':
    # data = torch.rand(100, 1, 3840)
    # train = MT_DCNN()
    # output = train(data)

    run_MT_DCNN()
