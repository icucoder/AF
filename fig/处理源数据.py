from train.AFDetectMain2 import *



def run_Process():
    begin = 20000
    # read_length = 10240
    read_length = 40960
    slidingWindowSize = 2048
    ECG_AF_vector, BCG_AF_vector, AF_persons, AF_label = get_AF_DataSet(begin, read_length, slidingWindowSize)
    ECG_NAF_vector, BCG_NAF_vector, NAF_persons, NAF_label = get_NAF_DataSet(begin, read_length, slidingWindowSize)

    torch.save(ECG_AF_vector, "../dataset/ECG_AF_vector.pth")
    torch.save(BCG_AF_vector, "../dataset/BCG_AF_vector.pth")
    torch.save(ECG_NAF_vector, "../dataset/ECG_NAF_vector.pth")
    torch.save(BCG_NAF_vector, "../dataset/BCG_NAF_vector.pth")
    print("数据保存完成")


if __name__ == '__main__':
    run_Process()