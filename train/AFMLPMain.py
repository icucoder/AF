import os
os.environ['TORCH_HOME'] = os.getcwd()

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


class ProcessedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1),
            nn.Linear(256, 64)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        mid = input.reshape(input.shape[0] * input.shape[1], 1, input.shape[-1])
        output = self.fc(mid)
        return output.reshape(input.shape[0], input.shape[1], output.shape[-1])


def train_Encoder(*, model, ecg_af, ecg_naf, bcg_af, bcg_naf, lr=0.001, epoch=2):
    criterion = nn.MSELoss()
    LossRecord = []
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
    dataset1 = TensorDataset(ecg_af, bcg_af)
    dataset2 = TensorDataset(ecg_naf, bcg_naf)
    data_loader1 = DataLoader(dataset=dataset1, batch_size=8, shuffle=True)
    data_loader2 = DataLoader(dataset=dataset2, batch_size=8, shuffle=True)
    for _ in tqdm(range(epoch)):
        for __, af_data_sample in enumerate(data_loader1, 1):
            for __, naf_data_sample in enumerate(data_loader2, 1):
                optimizer.zero_grad()
                loss1, loss2, loss3, loss4, loss5, loss6 = 0, 0, 0, 0, 0, 0
                # 获取数据
                ecg_af_mlp, bcg_af_mlp = af_data_sample
                ecg_naf_mlp, bcg_naf_mlp = naf_data_sample
                ecg_af_mlp = ecg_af_mlp.cuda()
                bcg_af_mlp = bcg_af_mlp.cuda()
                ecg_naf_mlp = ecg_naf_mlp.cuda()
                bcg_naf_mlp = bcg_naf_mlp.cuda()
                # 输入模型获取输出结果
                ecg_af_mlp_f = model(ecg_af_mlp)
                bcg_af_mlp_f = model(bcg_af_mlp)
                ecg_naf_mlp_f = model(ecg_naf_mlp)
                bcg_naf_mlp_f = model(bcg_naf_mlp)

                # 自对齐（连续性）
                loss1 += DataUtils.continuity_loss([ecg_af_mlp_f, bcg_af_mlp_f, ecg_naf_mlp_f, bcg_naf_mlp_f])
                # 互对齐
                loss2 += DataUtils.CLIP_loss(ecg_naf_mlp_f, ecg_af_mlp_f) + DataUtils.CLIP_loss(bcg_naf_mlp_f, bcg_af_mlp_f)
                # BCG方向与ECG方向对齐
                loss3 += criterion(DataUtils.CLIP_metric(ecg_naf_mlp_f, ecg_af_mlp_f), DataUtils.CLIP_metric(bcg_naf_mlp_f, bcg_af_mlp_f))
                # 按时间对齐提取到的特征
                loss4 += criterion(ecg_af_mlp_f, bcg_af_mlp_f) + criterion(ecg_naf_mlp_f, bcg_naf_mlp_f)
                # 添加margin
                margin = 10
                loss5 += DataUtils.MetricLoss(ecg_naf_mlp_f, ecg_af_mlp_f, margin) + DataUtils.MetricLoss(bcg_naf_mlp_f, bcg_af_mlp_f, margin)

                loss1 *= 1.0
                loss2 *= 1.0
                loss3 *= 1.0
                loss4 *= 1.0
                loss5 *= 100.0
                print(loss1, loss2, loss3, loss4, loss5)
                loss = loss1 + loss2 + loss3 + loss4 + loss5

                loss.backward()
                LossRecord.append(loss.item())
                optimizer.step()

        scheduler.step()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    plt.plot(LossRecord)
    plt.show()
    return model.cpu()


def get_dataset():
    ecg_af_mlp = torch.load("../output/ecg_af_mlp.pth", weights_only=False)
    ecg_naf_mlp = torch.load("../output/ecg_naf_mlp.pth", weights_only=False)
    bcg_af_mlp = torch.load("../output/bcg_af_mlp.pth", weights_only=False)
    bcg_naf_mlp = torch.load("../output/bcg_naf_mlp.pth", weights_only=False)
    return ecg_af_mlp, ecg_naf_mlp, bcg_af_mlp, bcg_naf_mlp


if __name__ == '__main__':
    ecg_af_mlp, ecg_naf_mlp, bcg_af_mlp, bcg_naf_mlp = get_dataset()

    model = ProcessedMLP()

    model = train_Encoder(
        model=model.cuda(),
        ecg_af=ecg_af_mlp.data,
        ecg_naf=ecg_naf_mlp.data,
        bcg_af=bcg_af_mlp.data,
        bcg_naf=bcg_naf_mlp.data,
        lr=0.003,
        epoch=1000
    )

    torch.save(model, "../model/ProcessedMLPModel.pth")
    print("训练结束，模型保存完成！")
