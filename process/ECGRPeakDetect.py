import numpy as np
from matplotlib import pyplot as plt



def inc_func(i, max_val):
    """
    自增函数
    """
    return i + 1 if i + 1 < max_val else 0


def dec_func(i, max_val):
    """
    自减函数
    """
    return i - 1 if i - 1 >= 0 else max_val - 1


def conv(data, coefficient):
    """
    卷积求和运算
    :param data: 一维数据
    :param coefficient: 系数
    :return:
    """
    temp, result = [data[0] for i in coefficient], []
    idx = 0
    for i in range(len(data)):
        temp[idx] = data[i]
        sum = 0
        for val in coefficient:
            sum += temp[idx] * val  # 卷积求和
            idx = dec_func(idx, len(coefficient))
        idx = inc_func(idx, len(coefficient))
        result.append(sum)

    return result


def derivative(data):
    """
    求导函数H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2))
    :param data: 数据
    :return: list
    """
    coef = [-1, -2, 0, 2, 1]
    der_data = conv(data, coef)
    return der_data



def moving_window_average(data, fs=200):
    """
    移动窗口积分均值
    y(nT) = (1/N)[x(nT-(N - 1)T)+ x(nT - (N - 2)T)+...+x(nT)]
    :param fs: 采样频率
    :param data: 数据
    :return: list
    """
    win_width = int(0.15 * fs + 0.5)  # 150ms的窗宽
    temp = conv(data, [1 for i in range(win_width)])
    res = [val / win_width for val in temp]
    return res


def findpeaks(data, min_distance, fs=200):
    """
    寻找峰值位置peak, 返回其峰值和位置
    :param fs: 采样率
    :param data: 数据
    :param min_distance: 两个峰值之间最小的距离, 0.1*fs
    :return: list, list
    """
    inc_cout = 0
    last_val = 0
    temp_val = 0
    temp_idx = 0
    peaks, locs = [], []
    for idx, val in enumerate(data):
        if inc_cout > 0:  # 如果已经开始计数了,便计数,防止一致是下降的曲线也进行统计
            inc_cout += 1
        if val > last_val and val > temp_val:
            temp_val = val
            temp_idx = idx
            inc_cout = 1
        elif val < (temp_val // 2) or inc_cout > min_distance:  # 下降到一半或者大于最小距离
            peaks.append(temp_val)
            locs.append(temp_idx)
            temp_val = 0
            inc_cout = 0
        last_val = val
    return peaks, locs

def find_max(vector):
    """
    找到一个数据中的最大值以及位置
    :param vector: 数据
    :return:
    """
    if len(vector) == 0:
        return 0, 0
    max_v, max_i = vector[0], 0
    for idx, val in enumerate(vector):
        if val > max_v:
            max_v = val
            max_i = idx
    return max_v, max_i

def mean(data):
    if len(data) == 0:
        raise '被除数为0'
    return sum(data) / len(data)

def diff(data):
    result = [0]
    for idx, val in enumerate(data[1:]):
        result.append(data[idx] - data[idx - 1])
    return result

def judge_rule(ecg_filter, ecg_win, peaks, locs, fs=200):
    """
    自适应阈值处理
    :param ecg_filter: list, 带通滤波后的数据
    :param ecg_win: list, 积分窗后的数据
    :param peaks: list, 通过积分窗后找到的峰值
    :param locs: list, peaks所对应的下标
    :param fs: int, 采样率
    :return: list, list, list, list
    """
    # 存错积分窗和带通滤波数据的信号
    qrs_amp_win = []  # 积分窗的qrs峰值
    qrs_idx_win = []  # 积分窗的qrs下标
    qrs_amp_flt = []  # bandpass filter的qrs峰值
    qrs_idx_flt = []  # bandpass filter的qrs下标
    # 信号和噪声
    thrs_win1, thrs_win2 = [], []  # 积分窗的高、低阈值
    thrs_flt1, thrs_flt2 = [], []  # 滤波的高、低阈值
    # 积分窗数据起始两秒的初始化，包括信号阈值和噪声阈值
    PEAKI, PEAKI_IDX = 0, 0
    THRESHOLD_I1 = 0.25 * max(ecg_win[:2 * fs])
    THRESHOLD_I2 = 0.5 * mean(ecg_win[:2 * fs])
    SPKI = THRESHOLD_I1
    NPKI = THRESHOLD_I2
    # 带通滤波数据起始两秒的初始化，包括信号阈值和噪声阈值
    PEAKF, PEAKF_IDX = 0, 0
    THRESHOLD_F1 = 0.25 * max(ecg_filter[:2 * fs])
    THRESHOLD_F2 = 0.5 * mean(ecg_filter[:2 * fs])
    SPKF = THRESHOLD_F1
    NPKF = THRESHOLD_F2
    # RR间期和均值
    RR_AVERAGE1 = 0
    RR_AVERAGE2 = 0
    rr_recent_limit = [0 for i in range(8)]  # 在限制范围内的RR间期
    rr_limit_idx, rr_limit_count = 0, 0  # 指针, 计数
    # 标志
    is_t_wave = False
    is_first_win = False
    is_new_qrs = False
    # 阈值自适应以及检测规则
    for i in range(len(peaks)):
        PEAKI, PEAKI_IDX = peaks[i], locs[i]  # 按照公式进行命名
        # 带通滤波数据中定位峰值, 在与peak有0.15秒的延迟范围内进行定位
        if PEAKI_IDX - int(0.15 * fs + 0.5) >= 1 and PEAKI_IDX <= len(ecg_filter):
            # 通过积分窗找到的peak,和原始数据之间的peak有延迟
            PEAKF, PEAKF_IDX = find_max(ecg_filter[PEAKI_IDX - int(0.15 * fs + 0.5):PEAKI_IDX])
        else:
            if i == 0:
                PEAKF, PEAKF_IDX = find_max(ecg_filter[:PEAKI_IDX])
                is_first_win = 1
            elif PEAKI_IDX >= len(ecg_filter):
                PEAKF, PEAKF_IDX = find_max(ecg_filter[PEAKI_IDX - int(0.15 * fs + 0.5):])

        # 更新心率
        if len(qrs_idx_win) >= 9:
            rr_interval = diff(qrs_idx_win[-9:])
            RR_AVERAGE1 = mean(rr_interval)
            latest_rr = qrs_idx_win[-1] - qrs_idx_win[-2]
            if rr_limit_count == 0:
                RR_AVERAGE2 = RR_AVERAGE1
            elif rr_limit_count < 8:
                RR_AVERAGE2 = mean(rr_recent_limit[:rr_limit_count])
            else:
                RR_AVERAGE2 = mean(rr_recent_limit)
            # 异常RR,更新阈值规则
            if latest_rr <= 0.92 * RR_AVERAGE2 or latest_rr >= 1.16 * RR_AVERAGE2:
                THRESHOLD_I1 *= 0.5
                THRESHOLD_F1 *= 0.5
            elif is_new_qrs:  # 正常RR, 且有新QRS波更新
                rr_recent_limit[rr_limit_idx] = latest_rr
                rr_limit_idx = inc_func(rr_limit_idx, 8)
                rr_limit_count += 1
                is_new_qrs = False

        # 回找策略
        if RR_AVERAGE2 != 0:  # 8秒后才会有值,所以前8秒不会回找,当然这里是可以优化的
            if locs[i] - qrs_idx_win[-1] >= int(1.66 * RR_AVERAGE2 + 0.5):  # 回找条件
                PEAKI, PEAKI_IDX = find_max(
                    ecg_win[qrs_idx_win[-1] + int(0.2 * fs + 0.5):locs[i] - int(0.2 * fs + 0.5)])
                # 将相对位置转换为绝对位置
                PEAKI_IDX = qrs_idx_win[-1] + int(0.2 * fs + 0.5) + PEAKI_IDX - 1
                if PEAKI > THRESHOLD_I2:  # 如果比低阈值高，则认为pks_temp为QRS
                    qrs_amp_win.append(PEAKI)
                    qrs_idx_win.append(PEAKI_IDX)
                    SPKI = 0.25 * PEAKI + 0.75 * SPKI  # 积分窗回找时SPKI的更新
                    is_new_qrs = True
                    # 定位带通滤波数据的QRS
                    if PEAKI_IDX <= len(ecg_filter):
                        find_range = ecg_filter[PEAKI_IDX - int(0.15 * fs + 0.5):PEAKI_IDX]
                    else:
                        find_range = ecg_filter[PEAKI_IDX - int(0.15 * fs + 0.5):]
                    PEAKF_B, PEAKF_B_IDX = find_max(find_range)  # search back, 滤波数据
                    # 是否满足带通滤波数据的阈值限制
                    if PEAKF_B > THRESHOLD_F2:
                        qrs_amp_flt.append(PEAKF_B)  # 保存滤数据中检出的R波
                        # 保存滤波数据中检出R波的下标
                        qrs_idx_flt.append(PEAKI_IDX - int(0.15 * fs + 0.5) + PEAKF_B_IDX)
                        SPKF = 0.25 * PEAKF_B + 0.75 * SPKF  # 滤波数据回找时SPKF的更新

        # 寻找噪声和QRS峰值
        if peaks[i] > THRESHOLD_I1:  # 满足高阈值
            if len(qrs_idx_win) >= 3:  # 最少3个R波，两个RR间期才能对比
                # 200ms到360ms范围内
                if int(0.20 * fs + 0.5) < locs[i] - qrs_idx_win[-1] <= int(0.36 * fs + 0.5):
                    cur_slope = mean(diff(ecg_win[locs[i] - int(0.075 * fs + 0.5):locs[i]]))
                    pre_slope = mean(
                        diff(ecg_win[qrs_idx_win[-1] - int(0.075 * fs + 0.5):qrs_idx_win[-1]]))
                    if abs(cur_slope) < 0.5 * abs(pre_slope):  # 检出T波的条件
                        PEAKI, PEAKI_IDX = peaks[i], locs[i]  # 防止被改，重新赋值
                        is_t_wave = True
                        # NPKI和NPKF规则更新
                        NPKF = 0.125 * PEAKF + 0.875 * NPKF
                        NPKI = 0.125 * PEAKI + 0.875 * NPKI
                    else:
                        is_t_wave = False

            # 正常情况下以及不是T波的时候
            if not is_t_wave:
                # 存储积分窗的QRS波峰和位置
                PEAKI, PEAKI_IDX = peaks[i], locs[i]  # 防止被改，重新赋值
                qrs_amp_win.append(PEAKI)
                qrs_idx_win.append(PEAKI_IDX)
                is_new_qrs = True
                SPKI = 0.125 * PEAKI + 0.875 * SPKI
                # 带通滤波检测阈值
                if PEAKF >= THRESHOLD_F2:
                    # 存储滤波后的QRS波峰和位置
                    if is_first_win:
                        qrs_idx_flt.append(PEAKF_IDX)
                    else:
                        qrs_idx_flt.append(PEAKI_IDX - int(0.15 * fs + 0.5) + PEAKF_IDX)
                    qrs_amp_flt.append(PEAKF)
                    SPKF = 0.125 * PEAKF + 0.875 * SPKF

        elif THRESHOLD_I2 <= PEAKI < THRESHOLD_I1:  # 在高阈值和低阈值之间
            NPKF = 0.125 * PEAKF + 0.875 * SPKF  # 滤波数据中, 噪声peak
            NPKI = 0.125 * PEAKI + 0.875 * SPKI  # 积分窗数据中, 噪声peak

        elif PEAKI < THRESHOLD_I2:  # 小于低阈值
            NPKF = 0.125 * PEAKF + 0.875 * SPKF  # 滤波数据中，噪声peak
            NPKI = 0.125 * PEAKI + 0.875 * SPKI  # 积分窗数据中, 噪声peak

        # 自适应信号和噪声阈值，即高、低阈值
        if NPKI != 0 or SPKI != 0:
            THRESHOLD_I1 = NPKI + 0.25 * abs(SPKI - NPKI)
            THRESHOLD_I2 = 0.5 * THRESHOLD_I1
        if NPKF != 0 or SPKF != 0:
            THRESHOLD_F1 = NPKF + 0.25 * abs(SPKF - NPKF)
            THRESHOLD_F2 = 0.5 * THRESHOLD_F1

        # 记录信号阈值(积分窗)
        thrs_win1.append(THRESHOLD_I1)
        thrs_win2.append(THRESHOLD_I2)
        # 记录信号阈值(滤波后)
        thrs_flt1.append(THRESHOLD_F1)
        thrs_flt2.append(THRESHOLD_F2)
        # 重置参数
        is_t_wave = False
        is_first_win = 0
    return qrs_amp_win, qrs_idx_win, qrs_amp_flt, qrs_idx_flt, thrs_win1, thrs_win2, thrs_flt1 \
        , thrs_flt2

def plot_peak_sig_and_noise_for_win_and_filter(data1, data2, x, y, x1, y1, locs1, th1, th2, th3,
                                               th4, title1='', title2=''):
    """
    绘制peak值的点
    :param data: 数据
    :param x: locs, 坐标位置
    :param y: peak值
    :return: None
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.title(title1)
    plt.plot(data1)
    plt.plot(x, y, '*r')
    plt.plot(locs1, th1)
    plt.plot(locs1, th2)
    plt.subplot(212)
    plt.title(title2)
    plt.plot(data2)
    plt.plot(x1, y1, 'og')
    plt.plot(locs1, th3)
    plt.plot(locs1, th4)
    plt.tight_layout()
    plt.show()




# 检测R峰算法，返回R峰的坐标点, 一般要求传入经过【5，15】Hz带通滤波过的ecg信号
# 返回原始数据、原始数据r峰坐标、滤波数据、滤波数据r峰坐标
def panTomkinsAlgorithm(originalEcg, filterEcg, fs=200):
    # 首先进行差分，因为QRS波斜率较高，使用增强其表示
    derivative_ecg = derivative(filterEcg)
    # 对信号进行平方处理
    pow_ecg = [val ** 2 for val in derivative_ecg]
    # 对信号进行积分
    integral_ecg = moving_window_average(pow_ecg, fs=fs)
    # 寻找基准peak
    peaks, locs = findpeaks(integral_ecg, min_distance=int(0.1 * fs))

    first_qrs_amp, first_qrs_idx, second_qrs_amp, second_qrs_idx, thrs_win1, thrs_win2, thrs_flt1, thrs_flt2 \
        = judge_rule(filterEcg, integral_ecg, peaks, locs, fs)

    # plot_peak_sig_and_noise_for_win_and_filter(integral_ecg, ecg, first_qrs_idx, first_qrs_amp,
    #                                                    second_qrs_idx,
    #                                                    second_qrs_amp, locs, thrs_win1, thrs_win2,
    #                                                    thrs_flt1, thrs_flt2, title1='Moving Window',
    #                                                    title2='Bandpass Filter')
    # 代表第一个判断依据下的R峰坐标（一般不用）；以及第二个判断依据下的R峰坐标
    originalQrsIndex = second_qrs_idx
    transferQrsIndex = second_qrs_idx.copy()
    # 根据变换空间上的坐标对原始信号上的坐标进行校准
    for index in range(0, len(originalQrsIndex)):
        needRevisedIndex = originalQrsIndex[index]
        boundary = int(fs * 0.05)
        new_max_index = needRevisedIndex - boundary + find_max(originalEcg[needRevisedIndex - boundary: needRevisedIndex + boundary])[1]
        originalQrsIndex[index] = new_max_index

    return originalEcg, np.array(originalQrsIndex), filterEcg, np.array(transferQrsIndex)


def rStandardPicture(originalEcg, originalQrsIndex, filterEcg, filterQrsIndex):
    plt.subplot(211)
    plt.plot(originalEcg)
    for index in originalQrsIndex:
        plt.plot(index, originalEcg[index], 'r*', markersize=15)  # 'r*' 表示红色的星号标记
    plt.legend()
    plt.subplot(212)
    plt.plot(filterEcg)
    for index in filterQrsIndex:
        plt.plot(index, filterEcg[index], 'r*', markersize=15)  # 'r*' 表示红色的星号标记
    plt.legend()
    plt.show()