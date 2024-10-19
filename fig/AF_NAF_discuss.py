from train.AFDetectMain2 import *
from utils.DataUtils import *
from sklearn.manifold import TSNE


def only_plot_2D_PCA_one_Figure(model, ECG_AF_vector, BCG_AF_vector, ECG_NAF_vector, BCG_NAF_vector, AF_index_list, NAF_index_list, AF_color_list, NAF_color_list):  # 输入数据形状shape：P, N, length
    assert len(AF_index_list) == len(AF_color_list)
    assert len(NAF_index_list) == len(NAF_color_list)

    _, _, sample, _, _, _ = model(ECG_AF_vector[0:1].detach(), BCG_AF_vector[0:1].detach())
    # 获取指定人的数据集
    ecg_af_mlp = torch.zeros(0, sample.shape[1], sample.shape[-1])
    for i in AF_index_list:
        _, _, af_mlp, _, _, _ = model(ECG_AF_vector[i:i + 1].detach(), BCG_AF_vector[i:i + 1].detach())
        ecg_af_mlp = torch.cat([ecg_af_mlp, af_mlp], dim=0)
    ecg_naf_mlp = torch.zeros(0, sample.shape[1], sample.shape[-1])
    for i in NAF_index_list:
        _, _, naf_mlp, _, _, _ = model(ECG_NAF_vector[i:i + 1].detach(), BCG_NAF_vector[i:i + 1].detach())
        ecg_naf_mlp = torch.cat([ecg_naf_mlp, naf_mlp], dim=0)

    processed_data_list = []  # 存放的数据形状shape：(P * N), length
    for data in [ecg_af_mlp, ecg_naf_mlp]:
        P, N, data_length = data.shape
        data = data.reshape(P * N, data_length)
        processed_data_list.append(data)
    all_data = torch.cat(processed_data_list, dim=0)

    pca = PCA(n_components=2, random_state=42)
    embedded_data = pca.fit_transform(all_data.detach().numpy())
    # tsne = TSNE(n_components=2, random_state=42)
    # embedded_data = tsne.fit_transform(all_data.detach().numpy())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(AF_index_list)):
        ax.scatter(
            embedded_data[i * sample.shape[1]:(i + 1) * sample.shape[1], 0],
            embedded_data[i * sample.shape[1]:(i + 1) * sample.shape[1], 1],
            c=AF_color_list[i], label='ECG AF'
        )
        for j in range(sample.shape[1]):
            ax.text(
                embedded_data[i * sample.shape[1] + j][0],
                embedded_data[i * sample.shape[1] + j][1],
                str(j),
                color=AF_color_list[i]
            )
    for k in range(len(NAF_index_list)):
        i = k + len(AF_index_list)
        ax.scatter(
            embedded_data[i * sample.shape[1]:(i + 1) * sample.shape[1], 0],
            embedded_data[i * sample.shape[1]:(i + 1) * sample.shape[1], 1],
            c=NAF_color_list[k], label='ECG NAF'
        )
        for j in range(sample.shape[1]):
            ax.text(
                embedded_data[i * sample.shape[1] + j][0],
                embedded_data[i * sample.shape[1] + j][1],
                str(j),
                color=NAF_color_list[k]
            )
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    ECG_AF_vector = torch.load("../dataset/ECG_AF_vector.pth")
    BCG_AF_vector = torch.load("../dataset/BCG_AF_vector.pth")
    ECG_NAF_vector = torch.load("../dataset/ECG_NAF_vector.pth")
    BCG_NAF_vector = torch.load("../dataset/BCG_NAF_vector.pth")

    model = torch.load("../model/ResNetModel.pth").eval()

    # AF_index_list = [1,2,3,4,5,6,7,8]
    # NAF_index_list = [1,2,3,4,5,6,7,8]
    # AF_color_list = ['r','y','r','y','r','y','r','y',]
    # NAF_color_list = ['b','black','b','black','b','black','b','black',]
    AF_index_list = [7, 8]
    NAF_index_list = [6, 10]
    AF_color_list = ['r', 'y', ]
    NAF_color_list = ['b', 'black', ]
    # _, _, ecg_naf_mlp, bcg_naf_mlp, _, _ = model(ECG_NAF_vector[begin_index:end_index].detach(), BCG_NAF_vector[begin_index:end_index].detach())
    # _, _, ecg_af_mlp, bcg_af_mlp, _, _ = model(ECG_AF_vector[begin_index:end_index].detach(), BCG_AF_vector[begin_index:end_index].detach())

    only_plot_2D_PCA_one_Figure(model, ECG_AF_vector, BCG_AF_vector, ECG_NAF_vector, BCG_NAF_vector, AF_index_list, NAF_index_list, AF_color_list, NAF_color_list)

    # 430  350

# 绘制指定index的TSNE降维分布图