import matplotlib.pyplot as plt
import torch
from utils.PltUtils import *
import seaborn as sns

import ECGRPeakDetect as tkE


# 计算RR间期箱线图
if __name__ == '__main__':
    ECG_AF_vector = torch.load("../dataset/ECG_AF_vector.pth")
    ECG_NAF_vector = torch.load("../dataset/ECG_NAF_vector.pth")

    # for i in range(ECG_AF_vector.shape[1]):
    #     ecg = ECG_AF_vector[7][i].detach()
    #     originalEcg, originalQrsIndex, filterEcg, filterQrsIndex = tkE.panTomkinsAlgorithm(ecg, ecg, fs=125)
    #     plt.plot(ecg)
    #     plt.plot(filterQrsIndex, ecg[filterQrsIndex], '*')
    #     plt.show()

    fig = plt.figure()
    plt.title("ECG NAF RR_interval")
    for i in range(ECG_AF_vector.shape[1]):
        ecg = ECG_NAF_vector[7][i].detach()

        originalEcg, originalQrsIndex, filterEcg, filterQrsIndex = tkE.panTomkinsAlgorithm(ecg, ecg, fs=125)

        RR_distance = ((filterQrsIndex[1:] - filterQrsIndex[:-1]) / 125.0)
        print(RR_distance)
        ax = fig.add_subplot(1, ECG_AF_vector.shape[1], i + 1)
        sns.boxplot(RR_distance)

        plt.ylim(0.5, 1.5)
        ax.set_yticks([0.5, 1.5], ['0.5s', '1.5s'])
    plt.show()

# 检测算法（返回原始数据、原始R峰、滤波数据、滤波数据R峰）(一般只用filterQrsIndex即可)


# originalEcg, originalQrsIndex, filterEcg, filterQrsIndex = tkE.panTomkinsAlgorithm(ecg, filterEcg)
# # 展示结果
# tkE.rStandardPicture(originalEcg, originalQrsIndex, filterEcg, filterQrsIndex)
