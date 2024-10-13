from unittest.mock import inplace

import torch.nn as nn
import torch


# 编码器的基本block，这里采用Resnet50/101/152的BigBasicBlock版本
# 三层卷积：第一层负责将in_channels->out_channels，第二层负责降采样，第三层负责将out_channels-> expansion * out_channels
# 这么做本质上是一个瓶颈结构，先扩大特征选择面，再通过第二层卷积滤除无效特征，最后通过第三层卷积被留下的特征
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downSampleStride=2, expansion=1):
        super().__init__()
        self.residual_block = nn.Sequential(
            # 第一个卷积用来调整输入通道数in_channels至out_channels
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # 降低样
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=downSampleStride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * expansion, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential()
        # 如果维度或者通道数发生了改变，那这里shortcut路径需要使用卷积调整
        if downSampleStride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * expansion, kernel_size=1, stride=downSampleStride, bias=False),
                nn.BatchNorm1d(out_channels * expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_block(x) + self.shortcut(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upSampleSize=2, upSampleStride=2):
        super().__init__()

        self.residual_block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=upSampleSize,stride=upSampleStride),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

        self.shortcut = nn.Sequential()
        # 如果维度或者通道数发生了改变，那这里shortcut路径需要使用卷积调整
        if upSampleSize != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size=upSampleSize, stride=upSampleStride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_block(x) + self.shortcut(x))



class ResNetAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = nn.Sequential(
            EncoderBlock(in_channels=1, out_channels=64, downSampleStride=2, expansion=1),
            EncoderBlock(in_channels=64, out_channels=128, downSampleStride=2, expansion=1),
            EncoderBlock(in_channels=128, out_channels=256, downSampleStride=2, expansion=1),
            EncoderBlock(in_channels=256, out_channels=512, downSampleStride=2, expansion=1),
        )

        self.Decoder = nn.Sequential(
            DecoderBlock(in_channels=512, out_channels=256, upSampleSize=2, upSampleStride=2),
            DecoderBlock(in_channels=256, out_channels=128, upSampleSize=2, upSampleStride=2),
            DecoderBlock(in_channels=128, out_channels=64, upSampleSize=2, upSampleStride=2),
            DecoderBlock(in_channels=64, out_channels=1, upSampleSize=2, upSampleStride=2),
        )

    def forward(self, x):
        mid_feature = self.Encoder(x)
        out = self.Decoder(mid_feature)
        return mid_feature, out


if __name__ == '__main__':
    a = torch.randn((10, 1, 128))
    model = ResNetAE()
    result1, result2 = model(a)
    print(result1.shape, result2.shape)
