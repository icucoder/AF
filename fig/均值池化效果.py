import matplotlib.pyplot as plt

from train.AFUnetMain1 import *
from utils.PltUtils import *

if __name__ == '__main__':
    # ECG_AF_vector = torch.load("../dataset/ECG_AF_vector.pth")
    ECG_AF_vector = torch.load("../dataset/ECG_NAF_vector.pth")
    # ECG_AF_vector = ECG_AF_vector - torch.mean(ECG_AF_vector, dim=-1).unsqueeze(2)

    kernel_size = 11
    avgpool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2, )

    trend = avgpool(ECG_AF_vector)
    print(ECG_AF_vector.shape, trend.shape)
    remain = ECG_AF_vector - trend

    for j in range(ECG_AF_vector.shape[0]):
        plt.suptitle("AvgPool1d kernel_size = " + str(kernel_size))
        for i in range(ECG_AF_vector.shape[1]):
            plt.subplot(ECG_AF_vector.shape[1], 2, 2 * i + 1)
            if i==0:
                plt.title("trend")
            plt.plot(trend[j][i].detach().numpy())
            plt.subplot(ECG_AF_vector.shape[1], 2, 2 * i + 2)
            if i==0:
                plt.title("series")
            plt.plot(remain[j][i].detach().numpy())
        plt.show()
