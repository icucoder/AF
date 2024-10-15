import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils import DataUtils
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


# def plot_2D_PCA_one_Figure(data_list):
#     fig = plt.figure()
#     colors = ['red', 'blue']
#     markers = ['*', 'o']
#     plt.title("red-AF / *-ECG")
#     for i in range(len(data_list)):
#         data = data_list[i]
#         data = data.reshape(data.shape[0] * data.shape[1], data.shape[-1])
#         data = DataUtils.get_PCA_feature(data.detach().numpy(), 3)
#         if i % 2 == 0:
#             ax = fig.add_subplot(1, len(data_list) // 2, i // 2 + 1, projection='3d')
#         # ax.plot(data[:, 0], data[:, 1], data[:, 2])
#         ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], marker=markers[i // 2], c=colors[i % 2])
#     plt.show()
#     return


def plot_2D_PCA_one_Figure(data_list):  # 输入数据形状shape：P, N, length
    length_list = [0]
    processed_data_list = []  # 存放的数据形状shape：(P * N), length
    for data in data_list:
        P, N, data_length = data.shape
        length_list.append(length_list[-1] + P * N)
        data = data.reshape(P * N, data_length)
        processed_data_list.append(data)
    all_data = torch.cat(processed_data_list, dim=0)

    # 应用 t-SNE 降维到二维空间
    # tsne = TSNE(n_components=2, random_state=42)
    # embedded_data = tsne.fit_transform(all_data.detach().numpy())
    # 应用 PCA 降维到二维空间
    pca = PCA(n_components=2, random_state=42)
    embedded_data = pca.fit_transform(all_data.detach().numpy())

    # 可视化
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # 定义颜色
    colors = ['r', 'g', 'b', 'y']
    label_names = ['AF_ECG', 'NAF_ECG', 'AF_BCG', 'NAF_BCG']
    fig_list = [121, 122]
    # 绘制每个类别的数据
    for i in range(len(data_list)):
        if i % 2 == 0:
            ax = fig.add_subplot(fig_list[i // 2])
        # ax = fig.add_subplot(111)
        ax.scatter(
            embedded_data[length_list[i]:length_list[i + 1], 0],
            embedded_data[length_list[i]:length_list[i + 1], 1],
            # embedded_data[length_list[i]:length_list[i + 1], 2],
            c=colors[i], label=label_names[i]
        )
        plt.legend()

    plt.title('t-SNE visualization of different classes')
    # ax.set_xlabel('Dimension 1')
    # ax.set_ylabel('Dimension 2')
    # ax.set_zlabel('Dimension 3')
    plt.show()
    return


def plot_3D_PCA_one_Figure(data_list):  # 输入数据形状shape：P, N, length
    length_list = [0]
    processed_data_list = []  # 存放的数据形状shape：(P * N), length
    for data in data_list:
        P, N, data_length = data.shape
        length_list.append(length_list[-1] + P * N)
        data = data.reshape(P * N, data_length)
        processed_data_list.append(data)
    all_data = torch.cat(processed_data_list, dim=0)

    # 应用 t-SNE 降维到二维空间
    # tsne = TSNE(n_components=2, random_state=42)
    # embedded_data = tsne.fit_transform(all_data.detach().numpy())
    # 应用 PCA 降维到二维空间
    pca = PCA(n_components=3, random_state=42)
    embedded_data = pca.fit_transform(all_data.detach().numpy())

    # 可视化
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # 定义颜色
    colors = ['r', 'g', 'b', 'y']
    label_names = ['AF_ECG', 'NAF_ECG', 'AF_BCG', 'NAF_BCG']

    # 绘制BCG、ECG到131中
    ax = fig.add_subplot(131, projection='3d')
    for i in range(len(data_list)):
        ax.scatter(
            embedded_data[length_list[i]:length_list[i + 1], 0],
            embedded_data[length_list[i]:length_list[i + 1], 1],
            embedded_data[length_list[i]:length_list[i + 1], 2],
            c=colors[i], label=label_names[i]
        )
        for j in range(length_list[i], length_list[i + 1]):
            ax.text(
                embedded_data[j][0],
                embedded_data[j][1],
                embedded_data[j][2],
                str(j),
            )
        plt.legend()


    # 分别绘制ECG、BCG到132、133两幅图中
    # 绘制每个类别的数据
    fig_list = [132, 133]
    for i in range(len(data_list)):
        if i % 2 == 0:
            ax = fig.add_subplot(fig_list[i // 2], projection='3d')
        # ax = fig.add_subplot(111)
        ax.scatter(
            embedded_data[length_list[i]:length_list[i + 1], 0],
            embedded_data[length_list[i]:length_list[i + 1], 1],
            embedded_data[length_list[i]:length_list[i + 1], 2],
            c=colors[i], label=label_names[i]
        )
        for j in range(length_list[i], length_list[i + 1]):
            ax.text(
                embedded_data[j][0],
                embedded_data[j][1],
                embedded_data[j][2],
                str(j),
            )
        plt.legend()

    plt.title('PCA visualization of different classes')
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
