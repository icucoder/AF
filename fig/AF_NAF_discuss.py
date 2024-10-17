from train.AFDetectMain2 import *
from utils.DataUtils import *
from utils import PltUtils

if __name__ == '__main__':
    ECG_AF_vector = torch.load("../dataset/ECG_AF_vector.pth")
    BCG_AF_vector = torch.load("../dataset/BCG_AF_vector.pth")
    ECG_NAF_vector = torch.load("../dataset/ECG_NAF_vector.pth")
    BCG_NAF_vector = torch.load("../dataset/BCG_NAF_vector.pth")

    model = torch.load("../model/ResNetModel.pth").eval()

    _, _, ecg_naf_mlp, bcg_naf_mlp, _, _ = model(ECG_NAF_vector[:1].detach(), BCG_NAF_vector[:1].detach())
    _, _, ecg_af_mlp, bcg_af_mlp, _, _ = model(ECG_AF_vector[:1].detach(), BCG_AF_vector[:1].detach())

    PltUtils.plot_2D_PCA_one_Figure([
        ecg_af_mlp,  # 有病的ECG
        ecg_naf_mlp,  # 无病的ECG
        bcg_af_mlp,  # 有病的BCG
        bcg_naf_mlp,  # 无病的BCG
    ])