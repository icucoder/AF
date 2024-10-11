from train.AFDetectMain2 import *

from utils import DataUtils, PltUtils

from utils import DataUtils, PltUtils

torch.manual_seed(10)

def get_AF_DataSet():
    ECGPathList = [
        '004.chengjinqing.20180319.171534.37.ecg.af.csv',
        '007.songjinming.20180320.174932.37.ecg.af.csv',
        '009.caidabao.20180321.180258.35.ecg.af.csv',
        '012.zhuyunlong.20180321.185039.38.ecg.af.csv',
        '027.wuxiangguan.20180326.175519.35.ecg.af.csv',
    ]
    BCGPathList = [
        '004.chengjinqing.20180319.171534.37.bcg.af.csv',
        '007.songjinming.20180320.174932.37.bcg.af.csv',
        '009.caidabao.20180321.180258.35.bcg.af.csv',
        '012.zhuyunlong.20180321.185039.38.bcg.af.csv',
        '027.wuxiangguan.20180326.175519.35.bcg.af.csv',
    ]

    for i in range(len(ECGPathList)):
        ECGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/ECG_cut/' + ECGPathList[i]
    for i in range(len(BCGPathList)):
        BCGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/BCG/' + BCGPathList[i]

    begin = 1000
    read_length = 10240
    slidingWindowSize = 2048
    ECG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    BCG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    for i in range(len(ECGPathList)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ECGPathList[i], begin=begin, length=read_length, column=2, isKansas=False)  # 标注采样hz 按时间读数据
        ECG_tmp = DataUtils.get_sliding_window_not_overlap(ECG, slidingWindowSize=slidingWindowSize).unsqueeze(0)
        ECG_vector = torch.cat([ECG_vector, ECG_tmp], dim=0)

        BCG = DataUtils.read_torch_from_CSV_data(path=BCGPathList[i], begin=begin, length=int(read_length * 1), column=2, isKansas=False)
        BCG = DataUtils.butter_bandpass_filter(BCG, 125, 1.0, 8.4)
        BCG_tmp = DataUtils.get_sliding_window_not_overlap(BCG, slidingWindowSize=slidingWindowSize).unsqueeze(0)
        BCG_vector = torch.cat([BCG_vector, BCG_tmp], dim=0)

    label = torch.zeros(ECG_vector.shape[0], 2, dtype=torch.float)
    onepersonnums = ECG_vector.shape[0] // len(ECGPathList)
    for i in range(len(ECGPathList)):
        label[i * onepersonnums:(i + 1) * onepersonnums, 1] = 1  # [0, 1]  AF
    ECG_vector = DataUtils.layernorm(ECG_vector)
    BCG_vector = DataUtils.layernorm(BCG_vector)
    return ECG_vector, BCG_vector, len(ECGPathList), label


def get_NAF_DataSet():
    ECGPathList = [
        '016.lijinliang.20180323.164358.36.ecg.na.csv',
        '017.liaoyinghua.20180323.162433.37.ecg.na.csv',
        '018.wangruihua.20180323.164452.35.ecg.na.csv',
        '020.shenwenbao.20180324.174851.35.ecg.na.csv',
    ]
    BCGPathList = [
        '016.lijinliang.20180323.164358.36.bcg.na.csv',
        '017.liaoyinghua.20180323.162433.37.bcg.na.csv',
        '018.wangruihua.20180323.164452.35.bcg.na.csv',
        '020.shenwenbao.20180324.174851.35.bcg.na.csv',
    ]

    for i in range(len(ECGPathList)):
        ECGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/ECG_cut/' + ECGPathList[i]
    for i in range(len(BCGPathList)):
        BCGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/BCG/' + BCGPathList[i]

    begin = 1000
    read_length = 10240
    slidingWindowSize = 2048
    ECG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    BCG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    for i in range(len(ECGPathList)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ECGPathList[i], begin=begin, length=read_length, column=2, isKansas=False)  # 标注采样hz 按时间读数据
        ECG_tmp = DataUtils.get_sliding_window_not_overlap(ECG, slidingWindowSize=slidingWindowSize).unsqueeze(0)
        ECG_vector = torch.cat([ECG_vector, ECG_tmp], dim=0)

        BCG = DataUtils.read_torch_from_CSV_data(path=BCGPathList[i], begin=begin, length=int(read_length * 1), column=2, isKansas=False)
        BCG = DataUtils.butter_bandpass_filter(BCG, 125, 1.0, 8.4)
        BCG_tmp = DataUtils.get_sliding_window_not_overlap(BCG, slidingWindowSize=slidingWindowSize).unsqueeze(0)
        BCG_vector = torch.cat([BCG_vector, BCG_tmp], dim=0)

    label = torch.zeros(ECG_vector.shape[0], 2, dtype=torch.float)
    onepersonnums = ECG_vector.shape[0] // len(ECGPathList)
    for i in range(len(ECGPathList)):
        label[i * onepersonnums:(i + 1) * onepersonnums, 0] = 1  # [1, 0]  NAF
    ECG_vector = DataUtils.layernorm(ECG_vector)
    BCG_vector = DataUtils.layernorm(BCG_vector)
    return ECG_vector, BCG_vector, len(ECGPathList), label


def run_Encoder():
    ECG_AF_vector, BCG_AF_vector, AF_persons, AF_label = get_AF_DataSet()
    ECG_NAF_vector, BCG_NAF_vector, NAF_persons, NAF_label = get_NAF_DataSet()

    ECG_vector = torch.cat([ECG_AF_vector, ECG_NAF_vector], dim=0)
    BCG_vector = torch.cat([BCG_AF_vector, BCG_NAF_vector], dim=0)
    label = torch.cat([AF_label, NAF_label], dim=0)

    print(ECG_vector.shape, BCG_vector.shape)
    # PltUtils.plot_all_data(ECG_vector)

    model = torch.load("../model/ResNetModel.pth").eval()

    ecg_af_feature, bcg_af_feature, ecg_af_mlp, bcg_af_mlp, ecg_af_restruct, bcg_af_restruct = model(ECG_AF_vector, BCG_AF_vector)
    ecg_naf_feature, bcg_naf_feature, ecg_naf_mlp, bcg_naf_mlp, ecg_naf_restruct, bcg_naf_restruct = model(ECG_NAF_vector, BCG_NAF_vector)
    print(ecg_af_feature.shape, bcg_af_feature.shape, ecg_af_mlp.shape, bcg_af_mlp.shape, ecg_af_restruct.shape, bcg_af_restruct.shape)


    PltUtils.plot_2D_PCA_one_Figure(
        [
            ecg_af_mlp,  # 有病的ECG
            ecg_naf_mlp,  # 无病的ECG
            bcg_af_mlp,  # 有病的BCG
            bcg_naf_mlp,  # 无病的BCG
        ]
    )



if __name__ == '__main__':
    run_Encoder()

# 自对齐：把每个人内部的NAF/AF特征排列成线性或曲线，即保证连续性
# 互对齐：将所有NAF聚集、将所有AF聚集
