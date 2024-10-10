from train.AFDetectMain import *

from utils import DataUtils, PltUtils

torch.manual_seed(10)


def get_test_DataSet():
    ECGPathList = [
        '004.chengjinqing.20180319.171534.37.ecg.af.csv',
        '007.songjinming.20180320.174932.37.ecg.af.csv',
        '009.caidabao.20180321.180258.35.ecg.af.csv',
        '012.zhuyunlong.20180321.185039.38.ecg.af.csv',
        '027.wuxiangguan.20180326.175519.35.ecg.af.csv',

        '001.wangqixiang.20180319.170246.38.ecg.af.csv',
        '002.zhufurong.20180319.164612.36.ecg.af.csv',
        '003.chenjinkang.20180319.170514.35.ecg.af.csv',
        '005.majialin.20180320.173223.35.ecg.af.csv',
        '006.wangxiangxinbao.20180320.174505.36.ecg.af.csv',
        '014.geyingdi.20180322.172818.37.ecg.af.csv',
        '016.lijinliang.20180323.164358.36.ecg.na.csv',
        '017.liaoyinghua.20180323.162433.37.ecg.na.csv',
        '018.wangruihua.20180323.164452.35.ecg.na.csv',
        '020.shenwenbao.20180324.174851.35.ecg.na.csv',
    ]
    BCGPathList = [
        '004.chengjinqing.20180319.171534.37.bcg.af.csv',
        '007.songjinming.20180320.174932.37.bcg.af.csv',
        '009.caidabao.20180321.180258.35.bcg.af.csv',
        '012.zhuyunlong.20180321.185039.38.bcg.af.csv',
        '027.wuxiangguan.20180326.175519.35.bcg.af.csv',

        '001.wangqixiang.20180319.170246.38.bcg.af.csv',
        '002.zhufurong.20180319.164612.36.bcg.af.csv',
        '003.chenjinkang.20180319.170514.35.bcg.af.csv',
        '005.majialin.20180320.173223.35.bcg.af.csv',
        '006.wangxiangxinbao.20180320.174505.36.bcg.af.csv',
        '014.geyingdi.20180322.172818.37.bcg.af.csv',
        '016.lijinliang.20180323.164358.36.bcg.na.csv',
        '017.liaoyinghua.20180323.162433.37.bcg.na.csv',
        '018.wangruihua.20180323.164452.35.bcg.na.csv',
        '020.shenwenbao.20180324.174851.35.bcg.na.csv',
    ]

    af_list = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(len(ECGPathList)):
        ECGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/ECG_cut/' + ECGPathList[i]
    for i in range(len(BCGPathList)):
        BCGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/BCG/' + BCGPathList[i]

    begin = 1000
    read_length = 10240
    slidingWindowSize = 2048
    ECG_vector = torch.zeros(0, 1, slidingWindowSize)
    BCG_vector = torch.zeros(0, 1, slidingWindowSize)
    for i in range(len(ECGPathList)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ECGPathList[i], begin=begin, length=read_length, column=2, isKansas=False)  # 标注采样hz 按时间读数据
        ECG_tmp = DataUtils.get_sliding_window_not_overlap(ECG, slidingWindowSize=slidingWindowSize).unsqueeze(1)
        ECG_vector = torch.cat([ECG_vector, ECG_tmp], dim=0)

        BCG = DataUtils.read_torch_from_CSV_data(path=BCGPathList[i], begin=begin, length=int(read_length * 1), column=2, isKansas=False)
        BCG = DataUtils.butter_bandpass_filter(BCG, 125, 1.0, 8.4)
        BCG_tmp = DataUtils.get_sliding_window_not_overlap(BCG, slidingWindowSize=slidingWindowSize).unsqueeze(1)
        BCG_vector = torch.cat([BCG_vector, BCG_tmp], dim=0)

    label = torch.zeros(ECG_vector.shape[0], 2, dtype=torch.float)
    onepersonnums = ECG_vector.shape[0] // len(ECGPathList)
    for i in range(len(af_list)):
        if (af_list[i] == 1):
            label[i * onepersonnums:(i + 1) * onepersonnums, 1] = 1  # [0, 1]  AF
        else:
            label[i * onepersonnums:(i + 1) * onepersonnums, 0] = 1  # [1, 0]  NAF
    ECG_vector = DataUtils.layernorm(ECG_vector)
    BCG_vector = DataUtils.layernorm(BCG_vector)
    return ECG_vector, BCG_vector, len(ECGPathList), label


def ttest_Encoder():
    ECG_vector, BCG_vector, persons, label = get_test_DataSet()

    model = torch.load("../model/ResNetModel.pth").eval()

    ecg_feature, bcg_feature, ecg_ans, bcg_ans, ecg_restruct, bcg_restruct = model(ECG_vector, BCG_vector)

    print("ECG分类结果", ecg_ans)
    print("BCG分类结果", bcg_ans)

    ecg_pca_feature = DataUtils.get_PCA_feature(ecg_feature.squeeze(1).detach().numpy(), 3)
    bcg_pca_feature = DataUtils.get_PCA_feature(bcg_feature.squeeze(1).detach().numpy(), 3)

    PltUtils.plot_3D_PCA_one_Figure(
        [
            ecg_pca_feature[:ecg_pca_feature.shape[0] // persons * 5],  # 有病的ECG
            ecg_pca_feature[ecg_pca_feature.shape[0] // persons * 5:],  # 无病的ECG
            bcg_pca_feature[:bcg_pca_feature.shape[0] // persons * 5],  # 有病的BCG
            bcg_pca_feature[bcg_pca_feature.shape[0] // persons * 5:],  # 无病的BCG
        ]
    )

    # 绘制特征是否对齐
    PltUtils.plot_bcg_ecg_feature(ecg_feature, bcg_feature)

    # 观察重构效果
    PltUtils.plot_origin_restruct_data(ECG_vector, ecg_restruct, BCG_vector, bcg_restruct)


if __name__ == '__main__':
    ttest_Encoder()
