import os
import matplotlib.pyplot as plt
from utils import DataUtils

if __name__ == '__main__':

    ecg_path = "H:/iScience/房颤数据/杭州原始数据/ECG_cut/"
    bcg_path = "H:/iScience/房颤数据/杭州原始数据/BCG/"

    # ecg_files = os.listdir(ecg_path)
    # bcg_files = os.listdir(bcg_path)

    ecg_files = [
        # '001.wangqixiang.20180319.170246.38.ecg.af.csv',
        # '002.zhufurong.20180319.164612.36.ecg.af.csv',
        # '003.chenjinkang.20180319.170514.35.ecg.af.csv',
        # '005.majialin.20180320.173223.35.ecg.af.csv',
        # '006.wangxiangxinbao.20180320.174505.36.ecg.af.csv',
        # '004.chengjinqing.20180319.171534.37.ecg.af.csv', '007.songjinming.20180320.174932.37.ecg.af.csv',
        # '009.caidabao.20180321.180258.35.ecg.af.csv', '012.zhuyunlong.20180321.185039.38.ecg.af.csv',
        # '027.wuxiangguan.20180326.175519.35.ecg.af.csv', '037.zhoudabao.20180412.175242.35.af.ecg.csv',
        # '040.shenlaiying.20180412.184414.38.af.ecg.csv', '043.zhangxiangzhen.20180413.184228.38.af.ecg.csv',
        # '047.zhengmeiying.20180416.193001.35.af.ecg.csv', '083.pinalin.20180612.204348.35.af.ecg.csv',
        # '091.wanqibao.20180614.205249.35.af.ecg.csv',
        '008.linlaiying.20180320.175323.38.ecg.na.csv', '013.yushuizhen.20180322.172202.36.ecg.na.csv',
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
        '078.yushuigen.20180424.192604.38.na.ecg.csv',
    ]

    bcg_files = [
        # '001.wangqixiang.20180319.170246.38.bcg.af.csv',
        # '002.zhufurong.20180319.164612.36.bcg.af.csv',
        # '003.chenjinkang.20180319.170514.35.bcg.af.csv',
        # '005.majialin.20180320.173223.35.bcg.af.csv',
        # '006.wangxiangxinbao.20180320.174505.36.bcg.af.csv',
        # '004.chengjinqing.20180319.171534.37.bcg.af.csv', '007.songjinming.20180320.174932.37.bcg.af.csv',
        # '009.caidabao.20180321.180258.35.bcg.af.csv', '012.zhuyunlong.20180321.185039.38.bcg.af.csv',
        # '027.wuxiangguan.20180326.175519.35.bcg.af.csv', '037.zhoudabao.20180412.175242.35.af.bcg.csv',
        # '040.shenlaiying.20180412.184414.38.af.bcg.csv', '043.zhangxiangzhen.20180413.184228.38.af.bcg.csv',
        # '047.zhengmeiying.20180416.193001.35.af.bcg.csv', '083.pinalin.20180612.204348.35.af.bcg.csv',
        # '091.wanqibao.20180614.205249.35.af.bcg.csv',
        '008.linlaiying.20180320.175323.38.bcg.na.csv', '013.yushuizhen.20180322.172202.36.bcg.na.csv',
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
        '078.yushuigen.20180424.192604.38.na.bcg.csv',
    ]

    assert len(ecg_files) == len(bcg_files)

    begin_list1 = [
        275000, 136000, 155900, 60000, 120000, 61000, 20000, 247000, 206000, 121000,
        188000,
    ]
    begin_list2 = [
        167100, 60000, 22000, 225000, 60000, 320000, 120000, 97000, 60000, 160000,
        93000, 165000, 103000, 160000, 110000, 82000, 54000, 135000, 86000, 30000,
        10000, 80000, 117000, 101000, 163000, 112000, 224000, 98000, 20000, 110000,
        50000,
    ]
    begin_list3 = [
        435000, 266000, 155900, 160000, 120000,
    ]
    read_length = 200000
    bias = 239
    for i in range(len(ecg_files)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ecg_path + ecg_files[i], begin=begin_list2[i], length=read_length, f=200, column=2)  # 标注采样hz 按时间读数据
        ECG = DataUtils.butter_bandpass_filter(ECG, 125, 1.0, 40.0)
        BCG = DataUtils.read_torch_from_CSV_data(path=bcg_path + bcg_files[i], begin=begin_list2[i] + bias, length=read_length, f=125, column=2)  # 标注采样hz 按时间读数据
        BCG = DataUtils.butter_bandpass_filter(BCG, 125, 1.0, 10.4)
        plt.subplot(211)
        plt.title(ecg_files[i])
        plt.plot(ECG)
        plt.subplot(212)
        plt.title(bcg_files[i])
        plt.plot(BCG)
        plt.show()