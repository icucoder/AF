from utils.PltUtils import *
import process.ECGRPeakDetect as tkE
from train.AFDetectMain2 import *

if __name__ == '__main__':
    ECG_AF_vector = torch.load("../dataset/ECG_AF_vector.pth")
    BCG_AF_vector = torch.load("../dataset/BCG_AF_vector.pth")
    ECG_NAF_vector = torch.load("../dataset/ECG_NAF_vector.pth")
    BCG_NAF_vector = torch.load("../dataset/BCG_NAF_vector.pth")
    ECG_UNKNOWN_vector = torch.load("../dataset/ECG_UNKNOWN_vector.pth")
    BCG_UNKNOWN_vector = torch.load("../dataset/BCG_UNKNOWN_vector.pth")

    # 绘制划分出的整晚房颤数据集源信号
    plot_origin_ecg_by_one_person_data_list([
        ECG_AF_vector[0:1], ECG_AF_vector[1:2], ECG_AF_vector[2:3], ECG_AF_vector[3:4], ECG_AF_vector[4:5]
    ],"AF")

    # 绘制划分出的非房颤数据集源信号
    plot_origin_ecg_by_one_person_data_list([
        ECG_NAF_vector[0:1], ECG_NAF_vector[1:2], ECG_NAF_vector[2:3], ECG_NAF_vector[3:4], ECG_NAF_vector[4:5]
    ], "NAF")

    # 绘制划分出的非持续性房颤数据集源信号
    plot_origin_ecg_by_one_person_data_list([
        ECG_UNKNOWN_vector[0:1], ECG_UNKNOWN_vector[1:2], ECG_UNKNOWN_vector[2:3], ECG_UNKNOWN_vector[3:4], ECG_UNKNOWN_vector[4:5]
    ], "UNKNOWN")
