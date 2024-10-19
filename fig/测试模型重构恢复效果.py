import matplotlib.pyplot as plt
import torch
from train.AFDetectMain2 import *


if __name__ == '__main__':
    ECG_AF_vector = torch.load("../dataset/ECG_AF_vector.pth")
    BCG_AF_vector = torch.load("../dataset/BCG_AF_vector.pth")
    ECG_NAF_vector = torch.load("../dataset/ECG_NAF_vector.pth")
    BCG_NAF_vector = torch.load("../dataset/BCG_NAF_vector.pth")

    model = torch.load("../model/ResNetModel.pth").eval()

    ecg_af_feature, bcg_af_feature, ecg_af_mlp, bcg_af_mlp, ecg_af_restruct, bcg_af_restruct = model(ECG_AF_vector, BCG_AF_vector)
    ecg_naf_feature, bcg_naf_feature, ecg_naf_mlp, bcg_naf_mlp, ecg_naf_restruct, bcg_naf_restruct = model(ECG_NAF_vector, BCG_NAF_vector)

    plt.subplot(211)
    plt.plot(ECG_AF_vector[0][0].detach().numpy())
    plt.subplot(212)
    plt.plot(ecg_af_restruct[0][0].detach().numpy())
    plt.show()

