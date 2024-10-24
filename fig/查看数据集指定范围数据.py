import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils import DataUtils

if __name__ == '__main__':
    ECG_AF_vector = torch.load("../dataset/ECG_AF_vector.pth")
    BCG_AF_vector = torch.load("../dataset/BCG_AF_vector.pth")
    ECG_NAF_vector = torch.load("../dataset/ECG_NAF_vector.pth")
    BCG_NAF_vector = torch.load("../dataset/BCG_NAF_vector.pth")
    ECG_UNKNOWN_vector = torch.load("../dataset/ECG_UNKNOWN_vector.pth")
    BCG_UNKNOWN_vector = torch.load("../dataset/BCG_UNKNOWN_vector.pth")

    vector = torch.cat([
        ECG_AF_vector.reshape(ECG_AF_vector.shape[0] * ECG_AF_vector.shape[1], ECG_AF_vector.shape[-1]),
        ECG_NAF_vector.reshape(ECG_NAF_vector.shape[0] * ECG_NAF_vector.shape[1], ECG_NAF_vector.shape[-1]),
        ECG_UNKNOWN_vector.reshape(ECG_UNKNOWN_vector.shape[0] * ECG_UNKNOWN_vector.shape[1], ECG_UNKNOWN_vector.shape[-1]),
    ], dim=0)

    print(vector.shape)

    # for i in [8, 13, 14, 15, 16, 17]:
    # for i in [666,670,667,696,488,811,812,815,]:
    for i in [203,205,202,1057,1058,218,1040,1047,]:
        plt.title(str(i))
        plt.plot(vector[i].detach().numpy())
        plt.show()
