import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from torch.nn import functional as F
import torch.nn as nn
from scipy.interpolate import interp1d


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
# f: 数据采样频率，读出数据后会统一转化为125Hz数据，输出的length统一按照125HZ计算
def read_torch_from_CSV_data(path, begin, length, f=125.0, column=1):
    data = read_numpy_from_CSV_data(path, int(begin * f / 125.0), int(length * f / 125.0),
                                    column)  # shape: (length * f / 125.0) * 1
    data = bilinear_interpolate_1d(data, length)  # 将读取的数据转化为125Hz
    return torch.from_numpy(data)


# 将形状为 M*1 的数据进行双线性插值到 N*1 的数据
def bilinear_interpolate_1d(data, N):
    M = data.shape[0]
    # 创建原始数据的 x 轴坐标
    x_original = np.linspace(0, 1, M)  # 原始数据的归一化坐标
    x_new = np.linspace(0, 1, N)  # 新的数据坐标
    # 插值函数
    interpolating_function = interp1d(x_original, data.flatten(), kind='linear')
    interpolated_data = interpolating_function(x_new)
    return interpolated_data.reshape(N, 1)


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
    return new_data.float()


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
        BCG = read_torch_from_CSV_data(path=BCGPathList[i], begin=0, length=int(read_length * 1), column=1,
                                       isKansas=True)
        BCG = butter_bandpass_filter(BCG, 125, 1.0, 8.4)
        ECG_tmp = get_sliding_window_overlap(ECG, slidingWindowSize=slidingWindowSize).unsqueeze(1)
        BCG_tmp = get_sliding_window_overlap(BCG, slidingWindowSize=slidingWindowSize).unsqueeze(1)
        ECG_vector = torch.cat([ECG_vector, ECG_tmp], dim=0)
        BCG_vector = torch.cat([BCG_vector, BCG_tmp], dim=0)
    return ECG_vector, BCG_vector, len(ECGPathList)


def layernorm(data):
    norm = nn.LayerNorm(data.shape[-1]).to(device=data.device)
    return norm(data)


def continuity_loss(vector_list):  # person num length
    ans = 0
    for vector in vector_list:
        diff = vector[:, 1:, :] - vector[:, :-1, :]
        diff = diff.reshape(diff.shape[0] * diff.shape[1], diff.shape[-1])
        ans += torch.sum(diff * diff)
    return ans


def CLIP_metric(naf_vector, af_vector):  # p n length
    naf_vector = naf_vector.reshape(naf_vector.shape[0] * naf_vector.shape[1], naf_vector.shape[-1])
    af_vector = af_vector.reshape(af_vector.shape[0] * af_vector.shape[1], af_vector.shape[-1])
    diff = af_vector.unsqueeze(1) - naf_vector.unsqueeze(0)
    diff = diff.reshape(diff.shape[0] * diff.shape[1], diff.shape[-1])
    diff = F.normalize(diff, dim=-1)
    return diff


def CLIP_loss(naf_vector, af_vector):  # p n length   后面减去前面
    diff = CLIP_metric(naf_vector, af_vector)
    ans = 1 - torch.mm(diff, diff.t())  # 两两向量之间的余弦相似度
    return torch.sum(ans ** 2)


def MetricLoss(naf_vector, af_vector, margin):  # p1 n1 length   p2 n2 length
    # 重构形状：(p1*n1) length  (p2*n2) length
    naf_vector = naf_vector.reshape(naf_vector.shape[0] * naf_vector.shape[1], naf_vector.shape[-1])
    af_vector = af_vector.reshape(af_vector.shape[0] * af_vector.shape[1], af_vector.shape[-1])
    # 使用广播做向量减法 输出形状： (p1*n1)*(p2*n2) length
    diff = af_vector.unsqueeze(1) - naf_vector.unsqueeze(0)
    diff = diff.reshape(diff.shape[0] * diff.shape[1], diff.shape[-1])
    # 计算向量长度  (p*n) 1
    distance = torch.sum(diff * diff, dim=1) ** 0.5
    # 计算loss:
    # if(distance>margin) loss += 0
    # if(distance<margin) loss += margin - distance
    relu = nn.ReLU()
    loss = torch.sum(relu(margin - distance))
    return loss


if __name__ == '__main__':
    data1 = torch.ones(5, 1, 10)
    data2 = torch.zeros(6, 1, 10)
    distance = MetricLoss(data1, data2, 5)
    print(distance)
