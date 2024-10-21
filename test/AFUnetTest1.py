from train.AFUnetMain1 import *
from utils.PltUtils import *


def plot_all_train_dataset():
    ecg_af_mlp = torch.load("../output/ecg_af_mlp.pth")
    ecg_naf_mlp = torch.load("../output/ecg_naf_mlp.pth")
    bcg_af_mlp = torch.load("../output/bcg_af_mlp.pth")
    bcg_naf_mlp = torch.load("../output/bcg_naf_mlp.pth")

    PltUtils.plot_2D_PCA_one_Figure([
        ecg_af_mlp,  # 有病的ECG
        ecg_naf_mlp,  # 无病的ECG
        bcg_af_mlp,  # 有病的BCG
        bcg_naf_mlp,  # 无病的BCG
    ])
    return


def plot_PCA_datalist_color_label():
    ecg_af_mlp = torch.load("../output/ecg_af_mlp.pth")
    ecg_naf_mlp = torch.load("../output/ecg_naf_mlp.pth")
    # bcg_af_mlp = torch.load("../output/bcg_af_mlp.pth")
    # bcg_naf_mlp = torch.load("../output/bcg_naf_mlp.pth")

    model = torch.load("../model/UnitedNetModel.pth").eval()


    # ECG_UNKNOWN_vector = torch.load("../dataset/ECG_UNKNOWN_vector.pth")
    # BCG_UNKNOWN_vector = torch.load("../dataset/BCG_UNKNOWN_vector.pth")
    # _, _, unknown_mlp, _ = model(ECG_UNKNOWN_vector[0:5].detach(), BCG_UNKNOWN_vector[0:5].detach())

    ECG_HUST_vector = torch.load("../dataset/ECG_HUST_vector.pth")
    BCG_HUST_vector = torch.load("../dataset/BCG_HUST_vector.pth")
    _, _, hust_mlp, _ = model(ECG_HUST_vector.detach(), BCG_HUST_vector.detach())

    # plot_2D_PCA_Figure_by_data_color_label(
    #     data_list=[ecg_af_mlp, ecg_naf_mlp, unknown_mlp[0:1], unknown_mlp[1:2], unknown_mlp[2:3], unknown_mlp[3:4], unknown_mlp[4:5]],
    #     colors=['r', 'g', 'y', '#FFC0CB', '#FFF0F5', '#DB7093', '#DA70D6', '#EE82EE'],
    #     label_names=['AF_ECG', 'NAF_ECG', 'UNKNOWN_ECG1', 'UNKNOWN_ECG2', 'UNKNOWN_ECG3', 'UNKNOWN_ECG4', 'UNKNOWN_ECG5'],
    # )

    plot_2D_PCA_Figure_by_data_color_label(
        data_list=[ecg_af_mlp, ecg_naf_mlp, hust_mlp[0:1], hust_mlp[1:2]],
        # data_list=[ecg_af_mlp, ecg_naf_mlp, ecg_af_mlp, ecg_naf_mlp],
        colors=['r', 'g', 'y', '#FFC0CB'],
        label_names=['AF_ECG', 'NAF_ECG', 'UNKNOWN_ECG1', 'UNKNOWN_ECG2'],
    )
    return


if __name__ == '__main__':
    plot_all_train_dataset()
    # plot_PCA_datalist_color_label()

