import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# @param data.shape: N * 3 (第3列为时间)
def plot_3D_PCA_Figure(data_list):
    fig = plt.figure()
    for i in range(len(data_list)):
        data = data_list[i]
        ax = fig.add_subplot(1, len(data_list), i + 1, projection='3d')
        jianbian = np.linspace(0, 1, data.shape[0])
        colors = plt.cm.RdYlBu(jianbian)
        ax.plot(data[:, 0], data[:, 1], data[:, 2])
        ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], marker='*', c=colors)
    plt.show()
    return


# @param data.shape: N * 3 (第3列为时间)
def plot_3D_PCA_one_Figure(data_list):
    fig = plt.figure()
    colors = ['red', 'blue']
    for i in range(len(data_list)):
        data = data_list[i]
        if i % 2 == 0:
            ax = fig.add_subplot(1, len(data_list) // 2, i // 2 + 1, projection='3d')
        # ax.plot(data[:, 0], data[:, 1], data[:, 2])
        ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], marker='*', c=colors[i % 2])
    plt.show()
    return


def plot_all_data(data):
    for i in range(data.shape[0]):
        plt.plot(data[i][0].detach().numpy())
    plt.show()


def plot_bcg_ecg_feature(ecg_feature, bcg_feature):
    ecg_diff = ecg_feature[1:] - ecg_feature[:-1]
    bcg_diff = bcg_feature[1:] - bcg_feature[:-1]
    num = min(ecg_diff.shape[0], bcg_diff.shape[0])
    for i in range(num):
        plt.title("ecg_feature and bcg_feature")
        plt.plot(ecg_feature[i][0].detach().numpy())
        plt.plot(bcg_feature[i][0].detach().numpy())
        plt.show()


def plot_origin_restruct_data(ecg_origin, ecg_restruct, bcg_origin, bcg_restruct):
    num = min(ecg_origin.shape[0], ecg_restruct.shape[0])
    for i in range(num):
        plt.subplot(211)
        plt.plot(ecg_origin[i][0].detach().numpy())
        plt.plot(ecg_restruct[i][0].detach().numpy())
        plt.subplot(212)
        plt.plot(bcg_origin[i][0].detach().numpy())
        plt.plot(bcg_restruct[i][0].detach().numpy())
        plt.show()
