import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ECGPathList = [
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
BCGPathList = [
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

for i in range(len(ECGPathList)):
    ECGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/ECG_cut/' + ECGPathList[i]
for i in range(len(BCGPathList)):
    BCGPathList[i] = 'H:/iScience/房颤数据/杭州原始数据/BCG/' + BCGPathList[i]

index = 2
begin = 2000
end = 40000
# bais = 1260 + 2580
bias = 1690
k = 1.6
# 008[18000-19250]
ecg = np.array(pd.read_csv(ECGPathList[index]))[int(begin*1.6):int(end*1.6), 1:2]
bcg = np.array(pd.read_csv(BCGPathList[index]))[begin+bias:end+bias, 1:2]
# ecg = np.array(pd.read_csv("H:/iScience/房颤数据/杭州原始数据/ECG_cut/008.linlaiying.20180320.175323.38.ecg.na.csv"))[int(begin*k):int(end*k), 1:2]
# bcg = np.array(pd.read_csv("H:/iScience/房颤数据/杭州原始数据/BCG/008.linlaiying.20180320.175323.38.bcg.na.csv"))[begin+bais:end+bais, 1:2]


plt.subplot(211)
plt.title("ecg")
plt.plot(ecg)
plt.subplot(212)
plt.title("bcg")
plt.plot(bcg)
plt.show()


# ecg = np.array(pd.read_csv("H:/iScience/房颤数据/杭州原始数据/ECG_cut/008.linlaiying.20180320.175323.38.ecg.na.csv"))
# bcg = np.array(pd.read_csv("H:/iScience/房颤数据/杭州原始数据/BCG/008.linlaiying.20180320.175323.38.bcg.na.csv"))
#
# print(ecg.shape[0]/200.0)
# print(bcg.shape[0]/125.0)
# print(ecg.shape[0]/200.0 - bcg.shape[0]/125.0)