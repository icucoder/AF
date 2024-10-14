import os
import matplotlib.pyplot as plt
from utils import DataUtils



if __name__ == '__main__':

    ecg_path = "H:/iScience/房颤数据/杭州原始数据/ECG_cut/"
    bcg_path = "H:/iScience/房颤数据/杭州原始数据/BCG/"

    ecg_files = os.listdir(ecg_path)
    bcg_files = os.listdir(bcg_path)

    assert len(ecg_files) == len(bcg_files)

    begin = 70000
    read_length = 5000
    bias = 1690
    for i in range(len(ecg_files)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ecg_path + ecg_files[i], begin=begin, length=read_length, f=200, column=2)  # 标注采样hz 按时间读数据
        BCG = DataUtils.read_torch_from_CSV_data(path=bcg_path + bcg_files[i], begin=begin + bias, length=read_length, f=125, column=2)  # 标注采样hz 按时间读数据

        plt.subplot(211)
        plt.title(ecg_files[i])
        plt.plot(ECG)
        plt.subplot(212)
        plt.title(bcg_files[i])
        plt.plot(BCG)
        plt.show()