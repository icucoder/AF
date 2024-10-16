import torch.nn as nn
import torch
from torchvision.models.optical_flow.raft import BottleneckBlock
from torchvision.models.resnet import BasicBlock


# 用于ResNet 18/34的中间层block，其在内部变化时通道数目不会发生改变(即in_channels要等于out_channels，这里不等也行)
class SmallBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # ResNet 18/34的基本residual_block仅有两个卷积组成
        self.residual_block = nn.Sequential(
            # 第一个卷积用来调整输入通道数in_channels至out_channels 与 下采样
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * SmallBasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * SmallBasicBlock.expansion),
        )
        self.shortcut = nn.Sequential()
        # 如果维度或者通道数发生了改变，那这里shortcut路径需要使用卷积调整
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * SmallBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * SmallBasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_block(x) + self.shortcut(x))

# 用于ResNet 50/101/152的中间层block，其在内部变化时通道数目会发生改变（最终输出的通道数目时out_channels的四倍）
class BigBasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # ResNet 50/101/152的基本residual_block有三个卷积组成
        self.residual_block = nn.Sequential(
            # 第一个卷积用来调整输入通道数in_channels至out_channels
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BigBasicBlock.expansion, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential()
        # 如果维度或者通道数发生了改变，那这里shortcut路径需要使用卷积调整
        if stride != 1 or in_channels != out_channels * BigBasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BigBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BigBasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_block(x) + self.shortcut(x))

# 主体框架
class ResNet(nn.Module):
    def __init__(self, in_channels, block, num_block, num_classes):
        super().__init__()
        # 原文layer1这里使用的是大小为7的卷积，卷积核过大，效果不好，因此这里修改成大小为3卷积
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        # 默认进入到conv2_x的卷积个数都为64,所以self.current_channels为64
        self.current_channels = 64
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(block, out_channels=64, num_blocks=num_block[0], stride=1)
        self.conv3 = self._make_layer(block, out_channels=128, num_blocks=num_block[1], stride=2)
        self.conv4 = self._make_layer(block, out_channels=256, num_blocks=num_block[2], stride=2)
        self.conv5 = self._make_layer(block, out_channels=512, num_blocks=num_block[3], stride=2)
        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    # 这里是用来产生中间层的block，其中block表示基本组成单元，由若干个（num_blocks个）smallBasicBlock或者bigBasicBlock组成
    # out_channels表示第一次需要调整的通道个数（最终basicBlock输出的通道数为这个的4倍），stride表示步长
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.current_channels, out_channels, stride))
            self.current_channels = out_channels * block.expansion # 1
        return nn.Sequential(*layers)


    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(self.maxPool(f1))
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(f4)
        print(f5.shape)
        output = self.softmax(self.fc(self.avgPool(f5).squeeze(-1)))
        return output

def resnet18(in_channels, num_classes):
    return ResNet(in_channels = in_channels, block = SmallBasicBlock, num_block = [2, 2, 2, 2], num_classes = num_classes)

def resnet34(in_channels, num_classes):
    return ResNet(in_channels = in_channels, block = SmallBasicBlock, num_block = [3, 4, 6, 3], num_classes = num_classes)

def resnet50(in_channels, num_classes):
    return ResNet(in_channels = in_channels, block = BigBasicBlock, num_block = [3, 4, 6, 3], num_classes = num_classes)

def resnet101(in_channels, num_classes):
    return ResNet(in_channels = in_channels, block = BigBasicBlock, num_block = [3, 4, 23, 3], num_classes = num_classes)

def resnet152(in_channels, num_classes):
    return ResNet(in_channels=in_channels, block = BigBasicBlock, num_block = [3, 8, 36, 3], num_classes = num_classes)


if __name__ == '__main__':
    # model = resnet18(in_channels = 1, num_classes = 2)
    model = resnet50(in_channels = 1, num_classes = 2)
    x = torch.randn(10, 1, 2048)
    print(model(x).shape)

