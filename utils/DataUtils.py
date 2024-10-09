import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from torch.nn import functional as F
import torch.nn as nn


def read_numpy_from_CSV_all_data(path):
    data = np.array(pd.read_csv(path))
    return data


# @param path: 文件路径
# @param begin: 读取数据起始范围
# @param length: 读取数据长度
# @param column: 读取数据列(column>=1, 默认为1)
# @return data.shape: length * 1
def read_numpy_from_CSV_data(path, begin, length, column=1):
    data = np.array(pd.read_csv(path))[begin:begin + length, column - 1:column]
    data = data.astype(np.float32)
    return data


# 同上
def read_torch_from_CSV_data(path, begin, length, column=1, isKansas=False):
    data = read_numpy_from_CSV_data(path, begin, length, column)
    if (isKansas):
        target_length = data.shape[0] // 8
        index = np.arange(target_length) * 8
        data = data[index]
    return torch.from_numpy(data)


# @param data.shape: N * 1
# @param slidingWindowSize: 特征采样滑动窗口大小
# @return new_data.shape: (N-slidingWindowSize+1) * slidingWindowSize
def get_sliding_window_overlap(data, slidingWindowSize):
    new_data = torch.zeros(0, slidingWindowSize)
    for i in range(0, data.shape[0] - slidingWindowSize - 1):
        new_data = torch.cat([new_data, data[i:i + slidingWindowSize, :].t()], dim=0)
    return new_data


def get_sliding_window_not_overlap(data, slidingWindowSize):
    new_data = torch.zeros(0, slidingWindowSize)
    i = 0
    while (i + slidingWindowSize <= data.shape[0]):
        new_data = torch.cat([new_data, data[i:i + slidingWindowSize, :].t()], dim=0)
        i = i + slidingWindowSize
    return new_data


# @param data.shape: N * dim
# @param target_dim: 保留主成分维数(0<target_dim<dim)
def get_PCA_feature(data, target_dim):
    assert 0 < target_dim < data.shape[-1]
    pca = PCA(n_components=target_dim)
    data = pca.fit_transform(data)
    return data


def butter_bandpass_filter(data, fs, lowcut=0.8, highcut=15, order=4):
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq

    # Design a Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='bandpass', analog=False)

    # Apply forward and reverse filtering using filtfilt for zero-phase response
    filtered_data = signal.filtfilt(b, a, data, axis=0)
    return torch.tensor(np.array(filtered_data), dtype=torch.float)


def savitzkyGolayFilter(data, window_size, polynomial_order):
    return savgol_filter(data, window_length=window_size, polyorder=polynomial_order)


def interpolation_from_BCG_to_ECG(data, ecg_bi_bcg):
    tmp = data.reshape(1, 1, data.shape[0])
    tmp = torch.from_numpy(tmp.copy())
    ans = F.interpolate(tmp, size=int(tmp.shape[-1] * ecg_bi_bcg), mode="linear")
    return ans.reshape(ans.shape[-1])


# @param data: numpy
def get_R_index_from_ECG(data):
    mean = torch.mean(data, dim=0)
    var = torch.std(data, dim=0)
    tmp = data.flatten()  # shape: N
    list = []
    for i in range(1, tmp.shape[0] - 2):
        if (tmp[i] > tmp[i - 1] and tmp[i] >= tmp[i + 1] and tmp[i] > (mean + 3 * var)):
            list.append(i)
    return torch.tensor(list)


def get_RR_diff_from_R_index(R_index_list, windows=5):
    RR_diff_list = torch.tensor(R_index_list[1:] - R_index_list[:-1], dtype=torch.float)
    hrv_list = []
    for i in range(RR_diff_list.shape[0] - 1 - windows):
        hrv = torch.std(RR_diff_list[i:i + windows])
        hrv_list.append(hrv)
    return hrv_list


def get_DataSet():
    ECGPathList = [
        'Original_ECG_X1001_290000.csv',
        'Original_ECG_X1002_380000.csv',
        'Original_ECG_X1003_340000.csv',
        'Original_ECG_X10036_350000.csv',

    ]
    BCGPathList = [
        'Original_BCG_X1001_290000.csv',
        'Original_BCG_X1002_380000.csv',
        'Original_BCG_X1003_340000.csv',
        'Original_BCG_X10036_350000.csv'
    ]
    for i in range(len(ECGPathList)):
        ECGPathList[i] = 'H:/iScience/房颤数据/Kansas房颤/' + ECGPathList[i]
        BCGPathList[i] = 'H:/iScience/房颤数据/Kansas房颤/' + BCGPathList[i]

    read_length = 250 * 8
    slidingWindowSize = 120
    ECG_vector = torch.zeros(0, 1, slidingWindowSize)
    BCG_vector = torch.zeros(0, 1, slidingWindowSize)
    for i in range(len(ECGPathList)):
        ECG = read_torch_from_CSV_data(path=ECGPathList[i], begin=0, length=read_length, column=1, isKansas=True)
        BCG = read_torch_from_CSV_data(path=BCGPathList[i], begin=0, length=int(read_length * 1), column=1, isKansas=True)
        BCG = butter_bandpass_filter(BCG, 125, 1.0, 8.4)
        ECG_tmp = get_sliding_window_overlap(ECG, slidingWindowSize=slidingWindowSize).unsqueeze(1)
        BCG_tmp = get_sliding_window_overlap(BCG, slidingWindowSize=slidingWindowSize).unsqueeze(1)
        ECG_vector = torch.cat([ECG_vector, ECG_tmp], dim=0)
        BCG_vector = torch.cat([BCG_vector, BCG_tmp], dim=0)
    return ECG_vector, BCG_vector, len(ECGPathList)


def layernorm(data):
    norm = nn.LayerNorm(data.shape[-1]).to(device=data.device)
    return norm(data)