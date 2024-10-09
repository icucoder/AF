import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from utils import DataUtils, PltUtils

torch.manual_seed(10)

class Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(256, 10000)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(10000, 2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, other):
        output = self.linear1(input)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.softmax(output)
        return output


def train_Encoder(*, model, ecg, bcg, label, lr=0.0001, epoch=2):
    criterion = nn.MSELoss()
    LossRecord = []
    for ep in tqdm(range(epoch)):
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
        optimizer.zero_grad()

        ecg_ans = model(ecg, bcg)
        loss4 = criterion(ecg_ans.squeeze(1), label)
        print(loss4)
        loss = loss4

        loss.backward()
        LossRecord.append(loss.item())
        optimizer.step()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    plt.plot(LossRecord)
    plt.show()
    return model


def get_DataSet():
    ECGPathList = [
        '004.chengjinqing.20180319.171534.37.ecg.af.csv',
        '007.songjinming.20180320.174932.37.ecg.af.csv',
        '009.caidabao.20180321.180258.35.ecg.af.csv',
        '012.zhuyunlong.20180321.185039.38.ecg.af.csv',
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
        '016.lijinliang.20180323.164358.36.bcg.na.csv',
        '017.liaoyinghua.20180323.162433.37.bcg.na.csv',
        '018.wangruihua.20180323.164452.35.bcg.na.csv',
        '020.shenwenbao.20180324.174851.35.bcg.na.csv',
    ]
    af_list = [
        1, 1, 1, 1, 0, 0, 0, 0
    ]
    for i in range(len(ECGPathList)):
        ECGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/ECG_cut/' + ECGPathList[i]
    for i in range(len(BCGPathList)):
        BCGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/BCG/' + BCGPathList[i]

    read_length = 1024
    slidingWindowSize = 256
    ECG_vector = torch.zeros(0, 1, slidingWindowSize)
    BCG_vector = torch.zeros(0, 1, slidingWindowSize)
    for i in range(len(ECGPathList)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ECGPathList[i], begin=1000, length=read_length, column=2, isKansas=False)
        BCG = DataUtils.read_torch_from_CSV_data(path=BCGPathList[i], begin=1000, length=int(read_length * 1), column=2, isKansas=False)
        BCG = DataUtils.butter_bandpass_filter(BCG, 125, 1.0, 8.4)
        ECG_tmp = DataUtils.get_sliding_window_not_overlap(ECG, slidingWindowSize=slidingWindowSize).unsqueeze(1)
        BCG_tmp = DataUtils.get_sliding_window_not_overlap(BCG, slidingWindowSize=slidingWindowSize).unsqueeze(1)
        ECG_vector = torch.cat([ECG_vector, ECG_tmp], dim=0)
        BCG_vector = torch.cat([BCG_vector, BCG_tmp], dim=0)
    label = torch.zeros(ECG_vector.shape[0], 2, dtype=torch.float)
    onepersonnums = ECG_vector.shape[0] // len(ECGPathList)
    for i in range(len(af_list)):
        if (af_list[i] == 1):
            label[i * onepersonnums:(i + 1) * onepersonnums, 1] = 1  # [0, 1]  AF
        else:
            label[i * onepersonnums:(i + 1) * onepersonnums, 0] = 1  # [1, 0]  NAF
    return ECG_vector, BCG_vector, len(ECGPathList), label


def run_Encoder():
    ECG_vector, BCG_vector, persons, label = get_DataSet()

    ECG_vector = DataUtils.layernorm(ECG_vector)
    BCG_vector = DataUtils.layernorm(BCG_vector)

    model = Classification()
    model = train_Encoder(model=model, ecg=ECG_vector.data, bcg=BCG_vector.data, label=label, lr=0.0001, epoch=200)

    class_ans = model(ECG_vector, BCG_vector)
    print("分类结果", class_ans)



if __name__ == '__main__':
    run_Encoder()
