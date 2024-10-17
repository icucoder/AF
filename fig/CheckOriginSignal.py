import os
import matplotlib.pyplot as plt
from utils import DataUtils

if __name__ == '__main__':

    ecg_path = "H:/iScience/房颤数据/杭州原始数据/ECG_cut/"
    bcg_path = "H:/iScience/房颤数据/杭州原始数据/BCG/"

    # ecg_files = os.listdir(ecg_path)
    # bcg_files = os.listdir(bcg_path)

    ecg_files = [
        '004.chengjinqing.20180319.171534.37.ecg.af.csv',
        '007.songjinming.20180320.174932.37.ecg.af.csv',
        '009.caidabao.20180321.180258.35.ecg.af.csv',
        '012.zhuyunlong.20180321.185039.38.ecg.af.csv',
        '027.wuxiangguan.20180326.175519.35.ecg.af.csv',
        '037.zhoudabao.20180412.175242.35.af.ecg.csv',
        '040.shenlaiying.20180412.184414.38.af.ecg.csv',
        '043.zhangxiangzhen.20180413.184228.38.af.ecg.csv',
        '047.zhengmeiying.20180416.193001.35.af.ecg.csv',
        '083.pinalin.20180612.204348.35.af.ecg.csv',
        '091.wanqibao.20180614.205249.35.af.ecg.csv',
    ]

    bcg_files = [
        '004.chengjinqing.20180319.171534.37.bcg.af.csv',
        '007.songjinming.20180320.174932.37.bcg.af.csv',
        '009.caidabao.20180321.180258.35.bcg.af.csv',
        '012.zhuyunlong.20180321.185039.38.bcg.af.csv',
        '027.wuxiangguan.20180326.175519.35.bcg.af.csv',
        '037.zhoudabao.20180412.175242.35.af.bcg.csv',
        '040.shenlaiying.20180412.184414.38.af.bcg.csv',
        '043.zhangxiangzhen.20180413.184228.38.af.bcg.csv',
        '047.zhengmeiying.20180416.193001.35.af.bcg.csv',
        '083.pinalin.20180612.204348.35.af.bcg.csv',
        '091.wanqibao.20180614.205249.35.af.bcg.csv',
    ]

    assert len(ecg_files) == len(bcg_files)

    begin_list1 = [
        275000, 136000, 155900, 60000, 120000, 61000, 20000, 247000, 206000, 121000,
        188000,
    ]
    begin_list2 = [
        280000, 60000, 20000, 20000, 20000, 50000, 70000, 20000, 20000, 20000,
        40000, 120000, 90000, 180000, 110000, 80000, 20000, 20000, 20000, 20000,
        20000, 80000, 20000, 20000, 30000, 80000, 20000, 20000, 20000, 45000,
        50000,
    ]
    read_length = 200000
    bias = 239
    for i in range(len(ecg_files)):
        ECG = DataUtils.read_torch_from_CSV_data(path=ecg_path + ecg_files[i], begin=begin_list1[i], length=read_length, f=200, column=2)  # 标注采样hz 按时间读数据
        ECG = DataUtils.butter_bandpass_filter(ECG, 125, 1.0, 40.0)
        BCG = DataUtils.read_torch_from_CSV_data(path=bcg_path + bcg_files[i], begin=begin_list1[i] + bias, length=read_length, f=125, column=2)  # 标注采样hz 按时间读数据
        BCG = DataUtils.butter_bandpass_filter(BCG, 125, 1.0, 10.4)
        plt.subplot(211)
        plt.title(ecg_files[i])
        plt.plot(ECG)
        plt.subplot(212)
        plt.title(bcg_files[i])
        plt.plot(BCG)
        plt.show()