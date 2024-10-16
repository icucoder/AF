from utils.PltUtils import *

data_root_path = 'H:/iScience/房颤数据/杭州原始数据/'


def get_AF_DataSet(begin_list, read_length, slidingWindowSize):
    ECGPathList = [
        '004.chengjinqing.20180319.171534.37.ecg.af.csv', '007.songjinming.20180320.174932.37.ecg.af.csv',
        '009.caidabao.20180321.180258.35.ecg.af.csv', '012.zhuyunlong.20180321.185039.38.ecg.af.csv',
        '027.wuxiangguan.20180326.175519.35.ecg.af.csv', '037.zhoudabao.20180412.175242.35.af.ecg.csv',
        '040.shenlaiying.20180412.184414.38.af.ecg.csv', '043.zhangxiangzhen.20180413.184228.38.af.ecg.csv',
        '047.zhengmeiying.20180416.193001.35.af.ecg.csv', '083.pinalin.20180612.204348.35.af.ecg.csv',
        '091.wanqibao.20180614.205249.35.af.ecg.csv',
    ]
    BCGPathList = [
        '004.chengjinqing.20180319.171534.37.bcg.af.csv', '007.songjinming.20180320.174932.37.bcg.af.csv',
        '009.caidabao.20180321.180258.35.bcg.af.csv', '012.zhuyunlong.20180321.185039.38.bcg.af.csv',
        '027.wuxiangguan.20180326.175519.35.bcg.af.csv', '037.zhoudabao.20180412.175242.35.af.bcg.csv',
        '040.shenlaiying.20180412.184414.38.af.bcg.csv', '043.zhangxiangzhen.20180413.184228.38.af.bcg.csv',
        '047.zhengmeiying.20180416.193001.35.af.bcg.csv', '083.pinalin.20180612.204348.35.af.bcg.csv',
        '091.wanqibao.20180614.205249.35.af.bcg.csv',
    ]

    for i in range(len(ECGPathList)):
        ECGPathList[i] = data_root_path + 'ECG_cut/' + ECGPathList[i]
    for i in range(len(BCGPathList)):
        BCGPathList[i] = data_root_path + 'BCG/' + BCGPathList[i]

    # begin = 1000
    # read_length = 10240
    # slidingWindowSize = 1024
    ECG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    BCG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    for i in range(len(ECGPathList)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ECGPathList[i], begin=begin_list[i], length=read_length, f=200,
                                                 column=2)  # 标注采样hz 按时间读数据
        ECG = DataUtils.butter_bandpass_filter(ECG, 125, 1.0, 8.4)
        ECG_tmp = DataUtils.get_sliding_window_not_overlap(ECG, slidingWindowSize=slidingWindowSize).unsqueeze(0)
        ECG_vector = torch.cat([ECG_vector, ECG_tmp], dim=0)

        BCG = DataUtils.read_torch_from_CSV_data(path=BCGPathList[i], begin=begin_list[i], length=read_length, f=125,
                                                 column=2)
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


def get_NAF_DataSet(begin_list, read_length, slidingWindowSize):
    ECGPathList = ['008.linlaiying.20180320.175323.38.ecg.na.csv', '013.yushuizhen.20180322.172202.36.ecg.na.csv',
                   '016.lijinliang.20180323.164358.36.ecg.na.csv', '017.liaoyinghua.20180323.162433.37.ecg.na.csv',
                   '018.wangruihua.20180323.164452.35.ecg.na.csv', '020.shenwenbao.20180324.174851.35.ecg.na.csv',
                   '021.sunjugen.20180324.181212.37.ecg.na.csv', '022.lincuiguan.20180324.180026.36.ecg.na.csv',
                   '023.wangzhaofa.20180325.175901.35.ecg.na.csv', '024.chengjinfang.20180325.182828.37.ecg.na.csv',
                   '025.chenrenxing.20180325.182125.36.ecg.na.csv', '026.shenying.20180326.181246.36.ecg.na.csv',
                   '028.luamei.20180326.182402.37.ecg.na.csv', '029.shichenhao.20180327.233252.36.ecg.na.csv',
                   '030.zhanghaiqiang.20180328.224655.36.ecg.na.csv', '031.yubin.20180329.191337.36.ecg.na.csv',
                   '045.chensuhua.20180414.180932.35.na.ecg.csv', '046.wujinhua.20180414.185039.37.na.ecg.csv',
                   '049.xiafurong.20180416.200429.37.na.ecg.csv', '053.yaoazhao.20180417.185423.37.na.ecg.csv',
                   '054.xufurong.20180417.190646.38.na.ecg.csv', '056.guyafu.20180418.191454.36.na.ecg.csv',
                   '057.wuhongde.20180418.185107.37.na.ecg.csv', '059.taoshouting.20180419.185644.35.na.ecg.csv',
                   '065.yusanjian.20180420.193147.37.na.ecg.csv', '069.geyongzhi.20180422.195719.37.na.ecg.csv',
                   '070.wuchuanyong.20180422.200924.38.na.ecg.csv', '072.xuliugen.20180423.193038.36.na.ecg.csv',
                   '075.panqijin.20180424.193717.35.na.ecg.csv', '077.wujinyu.20180424.195153.37.na.ecg.csv',
                   '078.yushuigen.20180424.192604.38.na.ecg.csv', ]
    BCGPathList = ['008.linlaiying.20180320.175323.38.bcg.na.csv', '013.yushuizhen.20180322.172202.36.bcg.na.csv',
                   '016.lijinliang.20180323.164358.36.bcg.na.csv', '017.liaoyinghua.20180323.162433.37.bcg.na.csv',
                   '018.wangruihua.20180323.164452.35.bcg.na.csv', '020.shenwenbao.20180324.174851.35.bcg.na.csv',
                   '021.sunjugen.20180324.181212.37.bcg.na.csv', '022.lincuiguan.20180324.180026.36.bcg.na.csv',
                   '023.wangzhaofa.20180325.175901.35.bcg.na.csv', '024.chengjinfang.20180325.182828.37.bcg.na.csv',
                   '025.chenrenxing.20180325.182125.36.bcg.na.csv', '026.shenying.20180326.181246.36.bcg.na.csv',
                   '028.luamei.20180326.182402.37.bcg.na.csv', '029.shichenhao.20180327.233252.36.bcg.na.csv',
                   '030.zhanghaiqiang.20180328.224655.36.bcg.na.csv', '031.yubin.20180329.191337.36.bcg.na.csv',
                   '045.chensuhua.20180414.180932.35.na.bcg.csv', '046.wujinhua.20180414.185039.37.na.bcg.csv',
                   '049.xiafurong.20180416.200429.37.na.bcg.csv', '053.yaoazhao.20180417.185423.37.na.bcg.csv',
                   '054.xufurong.20180417.190646.38.na.bcg.csv', '056.guyafu.20180418.191454.36.na.bcg.csv',
                   '057.wuhongde.20180418.185107.37.na.bcg.csv', '059.taoshouting.20180419.185644.35.na.bcg.csv',
                   '065.yusanjian.20180420.193147.37.na.bcg.csv', '069.geyongzhi.20180422.195719.37.na.bcg.csv',
                   '070.wuchuanyong.20180422.200924.38.na.bcg.csv', '072.xuliugen.20180423.193038.36.na.bcg.csv',
                   '075.panqijin.20180424.193717.35.na.bcg.csv', '077.wujinyu.20180424.195153.37.na.bcg.csv',
                   '078.yushuigen.20180424.192604.38.na.bcg.csv', ]

    for i in range(len(ECGPathList)):
        ECGPathList[i] = data_root_path + 'ECG_cut/' + ECGPathList[i]
    for i in range(len(BCGPathList)):
        BCGPathList[i] = data_root_path + 'BCG/' + BCGPathList[i]

    # begin = 1000
    # read_length = 10240
    # slidingWindowSize = 2048
    ECG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    BCG_vector = torch.zeros(0, read_length // slidingWindowSize, slidingWindowSize)
    for i in range(len(ECGPathList)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ECGPathList[i], begin=begin_list[i], length=read_length, f=200, column=2)  # 标注采样hz 按时间读数据
        ECG = DataUtils.butter_bandpass_filter(ECG, 125, 1.0, 40.0)
        ECG_tmp = DataUtils.get_sliding_window_not_overlap(ECG, slidingWindowSize=slidingWindowSize).unsqueeze(0)
        ECG_vector = torch.cat([ECG_vector, ECG_tmp], dim=0)

        BCG = DataUtils.read_torch_from_CSV_data(path=BCGPathList[i], begin=begin_list[i], length=read_length, f=125, column=2)
        BCG = DataUtils.butter_bandpass_filter(BCG, 125, 1.0, 10.4)
        BCG_tmp = DataUtils.get_sliding_window_not_overlap(BCG, slidingWindowSize=slidingWindowSize).unsqueeze(0)
        BCG_vector = torch.cat([BCG_vector, BCG_tmp], dim=0)

    label = torch.zeros(ECG_vector.shape[0], 2, dtype=torch.float)
    onepersonnums = ECG_vector.shape[0] // len(ECGPathList)
    for i in range(len(ECGPathList)):
        label[i * onepersonnums:(i + 1) * onepersonnums, 0] = 1  # [1, 0]  NAF
    ECG_vector = DataUtils.layernorm(ECG_vector)
    BCG_vector = DataUtils.layernorm(BCG_vector)
    return ECG_vector, BCG_vector, len(ECGPathList), label


def run_Process():
    begin_list1 = [
        95000, 105000, 20000, 60000, 20000, 130000, 20000, 80000, 20000, 150000, 80000]
    begin_list2 = [
        280000, 60000, 20000, 20000, 20000, 50000, 70000, 20000, 20000, 20000, 40000, 120000, 90000, 180000,
        110000, 80000, 20000, 20000, 20000, 20000, 20000, 80000, 20000, 20000, 30000, 80000, 20000, 20000, 20000, 45000,
        50000
    ]
    # read_length = 10240
    read_length = 40960
    slidingWindowSize = 2048
    ECG_AF_vector, BCG_AF_vector, AF_persons, AF_label = get_AF_DataSet(begin_list1, read_length, slidingWindowSize)
    ECG_NAF_vector, BCG_NAF_vector, NAF_persons, NAF_label = get_NAF_DataSet(begin_list2, read_length, slidingWindowSize)

    torch.save(ECG_AF_vector, "../dataset/ECG_AF_vector.pth")
    torch.save(BCG_AF_vector, "../dataset/BCG_AF_vector.pth")
    torch.save(ECG_NAF_vector, "../dataset/ECG_NAF_vector.pth")
    torch.save(BCG_NAF_vector, "../dataset/BCG_NAF_vector.pth")
    print("数据保存完成")


def plot_origin_process_data():
    ECG_AF_vector = torch.load("../dataset/ECG_AF_vector.pth")
    BCG_AF_vector = torch.load("../dataset/BCG_AF_vector.pth")
    ECG_NAF_vector = torch.load("../dataset/ECG_NAF_vector.pth")
    BCG_NAF_vector = torch.load("../dataset/BCG_NAF_vector.pth")

    for i in range(ECG_AF_vector.shape[0]):
        for j in range(ECG_AF_vector.shape[1]):
            plt.subplot(ECG_AF_vector.shape[1], 2, j * 2 + 1)
            if j == 0: plt.title("AF (ECG left) (BCG right)")
            plt.plot(ECG_AF_vector[i][j].detach().numpy())
            plt.subplot(ECG_AF_vector.shape[1], 2, j * 2 + 2)
            plt.plot(BCG_AF_vector[i][j].detach().numpy())
        plt.show()

    for i in range(ECG_NAF_vector.shape[0]):
        for j in range(ECG_NAF_vector.shape[1]):
            plt.subplot(ECG_NAF_vector.shape[1], 2, j * 2 + 1)
            if j == 0: plt.title("NAF (ECG left) (BCG right)")
            plt.plot(ECG_NAF_vector[i][j].detach().numpy())
            plt.subplot(ECG_NAF_vector.shape[1], 2, j * 2 + 2)
            plt.plot(BCG_NAF_vector[i][j].detach().numpy())
        plt.show()


if __name__ == '__main__':
    # run_Process() # 生成源数据集
    plot_origin_process_data()  # 绘制源数据集
