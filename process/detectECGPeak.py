import matplotlib.pyplot as plt
import torch
from utils.PltUtils import *
import seaborn as sns

import ECGRPeakDetect as tkE

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
    plt.title("ECG NAF bpm")
    for i in range(ECG_AF_vector.shape[1]):
        ecg = ECG_NAF_vector[7][i].detach()

        originalEcg, originalQrsIndex, filterEcg, filterQrsIndex = tkE.panTomkinsAlgorithm(ecg, ecg, fs=125)

        RR_distance = ((filterQrsIndex[1:] - filterQrsIndex[:-1]) / 125.0)
        print(RR_distance)
        ax = fig.add_subplot(1, ECG_AF_vector.shape[1], i + 1)
        sns.boxplot(RR_distance)
        # plt.ylim(0.5, 1.5)
        # ax.set_yticks([0.5, 1.5], ['0.5s', '1.5s'])

        plt.ylim(0.5, 1.5)
        ax.set_yticks([0.5, 1.5], ['0.5s', '1.5s'])

        # plt.plot(ecg)
        # plt.plot(filterQrsIndex, ecg[filterQrsIndex], '*')
    plt.show()

# read_csv
# ecg = np.array(pd.read_csv('D:\\ECG_BCG\\ECG\\001.wangqixiang.20180319.170246.38.ecg.af.csv',skiprows=1))[500000:600000,1:2].squeeze(axis=1)
# ecg = np.array(pd.read_csv('/Users/cathy/Desktop/数据/房颤BCG_ECG/ECG/001.wangqixiang.20180319.170246.38.ecg.af.csv',skiprows=1))[500000:500500,1:2].squeeze(axis=1)
# ecg = np.array(pd.read_csv('/Users/cathy/Desktop/数据/房颤BCG_ECG/ECG/001.wangqixiang.20180319.170246.38.ecg.af.csv', skiprows=1))[500000:520000,1:2].squeeze(axis=1)

# 进行滤波
# filterEcg = BU.zeroPhaseButterWorthFilter(data=ecg, lowcut=5.0, highcut=15.0, fs=200, order=3)


# 检测算法（返回原始数据、原始R峰、滤波数据、滤波数据R峰）(一般只用filterQrsIndex即可)


# originalEcg, originalQrsIndex, filterEcg, filterQrsIndex = tkE.panTomkinsAlgorithm(ecg, filterEcg)
# # 展示结果
# tkE.rStandardPicture(originalEcg, originalQrsIndex, filterEcg, filterQrsIndex)
