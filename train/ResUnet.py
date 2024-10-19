
import torch.nn as nn
import torch


# 编码器的基本block，这里采用Resnet50/101/152的BigBasicBlock版本
# 三层卷积：第一层负责将in_channels->out_channels，第二层负责降采样，第三层负责将out_channels-> expansion * out_channels
# 这么做本质上是一个瓶颈结构，先扩大特征选择面，再通过第二层卷积滤除无效特征，最后通过第三层卷积被留下的特征
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.shortcut = nn.Sequential()
        # 如果维度或者通道数发生了改变，那这里shortcut路径需要使用卷积调整
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_block(x) + self.shortcut(x))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upSampleSize=2, upSampleStride=2):
        super().__init__()

        self.residual_block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=upSampleSize, stride=upSampleStride),
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



class ResUnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.Encoder1 = EncoderBlock(in_channels=1, out_channels=64)
        self.Encoder2 = EncoderBlock(in_channels=64, out_channels=128)
        self.Encoder3 = EncoderBlock(in_channels=128, out_channels=256)
        self.Encoder4 = EncoderBlock(in_channels=256, out_channels=512)
        self.BottleEncoder = EncoderBlock(in_channels=512, out_channels=1024)

        self.downSample = nn.MaxPool1d(kernel_size=2)

        self.upSampleBlock4 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.Decoder4 = EncoderBlock(in_channels=1024, out_channels=512)

        self.upSampleBlock3 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.Decoder3 = EncoderBlock(in_channels=512, out_channels=256)

        self.upSampleBlock2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.Decoder2 = EncoderBlock(in_channels=256, out_channels=128)

        self.upSampleBlock1 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.Decoder1 = EncoderBlock(in_channels=128, out_channels=64)

        self.toOutput = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        down1_conn = self.Encoder1(x)
        down1 = self.downSample(down1_conn)

        down2_conn = self.Encoder2(down1)
        down2 = self.downSample(down2_conn)

        down3_conn = self.Encoder3(down2)
        down3 = self.downSample(down3_conn)

        down4_conn = self.Encoder4(down3)
        down4 = self.downSample(down4_conn)

        mid_out = self.BottleEncoder(down4)

        up5 = self.Decoder4(torch.cat([down4_conn, self.upSampleBlock4(mid_out)],dim=1))
        up4 = self.Decoder3(torch.cat([down3_conn, self.upSampleBlock3(up5)], dim=1))
        up3 = self.Decoder2(torch.cat([down2_conn, self.upSampleBlock2(up4)], dim=1))
        up2 = self.Decoder1(torch.cat([down1_conn, self.upSampleBlock1(up3)], dim=1))
        reConstruct = self.toOutput(up2)
        return reConstruct, down1_conn, down2_conn, down3_conn, down4_conn, mid_out


if __name__ == '__main__':
    model = ResUnet()
    data = torch.randn((10, 1, 256))
    reConstruct, down1_conn, down2_conn, down3_conn, down4_conn, mid_out = model(data)
    print(reConstruct.shape)
    print(down1_conn.shape)
    print(down2_conn.shape)
    print(down3_conn.shape)
    print(down4_conn.shape)
    print(mid_out.shape)