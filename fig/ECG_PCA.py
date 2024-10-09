import matplotlib.pyplot as plt
import torch

from utils import DataUtils
from utils import PltUtils

if __name__ == '__main__':
    length = 280000
    # length * 1
    # ECG = DataUtils.read_torch_from_CSV_data(path='H:/iScience/房颤数据/杭州房颤/AF-ECG001.csv', begin=0, length=length, column=1)
    # BCG1 = DataUtils.read_torch_from_CSV_data(path='H:/iScience/房颤数据/杭州房颤/AF-BCG001.csv', begin=10, length=int(length * 5 / 8), column=1)
    # ECG = DataUtils.read_torch_from_CSV_data(path='H:/iScience/房颤数据/Kansas房颤/Original_ECG_X1001_290000.csv', begin=10, length=length, column=1)
    # BCG1 = DataUtils.read_torch_from_CSV_data(path='H:/iScience/房颤数据/Kansas房颤/Original_BCG_X1001_290000.csv', begin=10, length=int(length * 1), column=1)
    ECG = DataUtils.read_torch_from_CSV_data(path='H:/iScience/房颤数据/Kansas房颤/Original_ECG_X10036_350000.csv', begin=10, length=length, column=1)
    BCG1 = DataUtils.read_torch_from_CSV_data(path='H:/iScience/房颤数据/Kansas房颤/Original_BCG_X10036_350000.csv', begin=10, length=int(length * 1), column=1)
    BCG2 = DataUtils.butter_bandpass_filter(BCG1, 1000, 1.0, 8.4)
    # ECG_vector = DataUtils.get_sliding_window(ECG, slidingWindowSize=21)
    # print(ECG_vector.shape)
    # ECG_feature = DataUtils.get_PCA_feature(ECG_vector, target_dim=2)  # 获取主成分向量
    # ECG_feature = torch.cat([torch.from_numpy(ECG_feature), torch.arange(ECG_feature.shape[0], dtype=torch.float).reshape(ECG_feature.shape[0], 1)], dim=1)  # 添加时间列

    # 绘制3d特征图
    # PltUtils.plot_3D_PCA_Figure([ECG_feature])

    # RR间期
    R_index_list = DataUtils.get_R_index_from_ECG(ECG)
    hrv_list = DataUtils.get_RR_diff_from_R_index(R_index_list, windows=10)
    print(hrv_list)


    # 绘制原始图像
    plt.subplot(411)
    plt.title("ECG")
    plt.plot(ECG)
    plt.subplot(412)
    plt.title("BCG ECG")
    t = torch.arange(ECG.shape[0], dtype=torch.float)/1000 # 单位/秒
    plt.plot(t, DataUtils.interpolation_from_BCG_to_ECG(BCG2, 1)*50)
    plt.plot(t, ECG)
    plt.plot(R_index_list/1000.0, ECG[R_index_list], '*')
    plt.subplot(413)
    plt.title("BCG after bandpass")
    plt.plot(BCG2)
    plt.subplot(414)
    plt.title("BCG")
    plt.plot(BCG1)
    plt.show()

    # ECG1 = DataUtils.read_numpy_from_CSV_all_data('H:/iScience/房颤数据/杭州房颤/AF-ECG001.csv')
    # BCG1 = DataUtils.read_numpy_from_CSV_all_data('H:/iScience/房颤数据/杭州房颤/AF-BCG001.csv')
    # print(ECG1.shape, BCG1.shape)
