from train.AFMLPMain import *

from utils import DataUtils, PltUtils

if __name__ == '__main__':
    ecg_af_mlp, ecg_naf_mlp, bcg_af_mlp, bcg_naf_mlp = get_dataset()

    model = torch.load("../model/ProcessedMLPModel.pth").eval()

    # 输入模型获取输出结果
    ecg_af_mlp_f = model(ecg_af_mlp)
    bcg_af_mlp_f = model(bcg_af_mlp)
    ecg_naf_mlp_f = model(ecg_naf_mlp)
    bcg_naf_mlp_f = model(bcg_naf_mlp)

    PltUtils.plot_2D_PCA_one_Figure([
        ecg_af_mlp_f,  # 有病的ECG
        ecg_naf_mlp_f,  # 无病的ECG
        bcg_af_mlp_f,  # 有病的BCG
        bcg_naf_mlp_f,  # 无病的BCG
    ])

    print(torch.__file__)