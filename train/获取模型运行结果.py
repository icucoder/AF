from AFUnetMain1 import *

if __name__ == '__main__':
    ECG_AF_vector = torch.load("../dataset/ECG_AF_vector.pth")
    BCG_AF_vector = torch.load("../dataset/BCG_AF_vector.pth")
    ECG_NAF_vector = torch.load("../dataset/ECG_NAF_vector.pth")
    BCG_NAF_vector = torch.load("../dataset/BCG_NAF_vector.pth")

    model = torch.load("../model/UnitedNetModel.pth").eval()
    model = model.eval().cuda()
    _, _, ecg_af_mlp, bcg_af_mlp = model(ECG_AF_vector[0:1, :, :].cuda(), BCG_AF_vector[0:1, :, :].cuda())
    for i in range(1, ECG_AF_vector.shape[0]):
        _, _, ecg_af_mlp_mid, bcg_af_mlp_mid = model(ECG_AF_vector[i:i + 1, :, :].cuda(), BCG_AF_vector[i:i + 1, :, :].cuda())
        ecg_af_mlp = torch.cat([ecg_af_mlp.detach(), ecg_af_mlp_mid.detach()], dim=0)
        bcg_af_mlp = torch.cat([bcg_af_mlp.detach(), bcg_af_mlp_mid.detach()], dim=0)

    _, _, ecg_naf_mlp, bcg_naf_mlp = model(ECG_NAF_vector[0:1, :, :].cuda(), BCG_NAF_vector[0:1, :, :].cuda())
    for i in range(1, ECG_NAF_vector.shape[0]):
        _, _, ecg_naf_mlp_mid, bcg_naf_mlp_mid = model(ECG_NAF_vector[i:i + 1, :, :].cuda(), BCG_NAF_vector[i:i + 1, :, :].cuda())
        ecg_naf_mlp = torch.cat([ecg_naf_mlp.detach(), ecg_naf_mlp_mid.detach()], dim=0)
        bcg_naf_mlp = torch.cat([bcg_naf_mlp.detach(), bcg_naf_mlp_mid.detach()], dim=0)

    print(ecg_af_mlp.shape, bcg_af_mlp.shape, ecg_naf_mlp.shape, bcg_naf_mlp.shape)
    torch.save(ecg_af_mlp.detach().cpu(), "../output/ecg_af_mlp.pth")
    torch.save(bcg_af_mlp.detach().cpu(), "../output/bcg_af_mlp.pth")
    torch.save(ecg_naf_mlp.detach().cpu(), "../output/ecg_naf_mlp.pth")
    torch.save(bcg_naf_mlp.detach().cpu(), "../output/bcg_naf_mlp.pth")
    print("结果保存完成！")