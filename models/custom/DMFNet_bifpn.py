"""
This model adds MFUnit into each Residual Path, to make the gradient easier in learning. (idea from Unet++ paper)
"""

import torch
from torch import nn

try:
    from models.sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass

from models.DMFNet_16x import normalization, Conv3d_Block, DilatedConv3DBlock, MFunit, DMFUnit


class DMFNet_bifpn(nn.Module):
    def __init__(self, c=4, n=32, channels=128, groups=16, norm='bn', num_classes=4):
        super(DMFNet_bifpn, self).__init__()

        # Entry flow
        self.encoder_block1 = nn.Conv3d(c, n, kernel_size=3, padding=1, stride=2, bias=False)  # H//2
        self.encoder_block2 = nn.Sequential(
            DMFUnit(n, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block3 = nn.Sequential(
            DMFUnit(channels, channels * 2, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block4 = nn.Sequential(  # H//8,channels*4
            MFunit(channels * 2, channels * 3, g=groups, stride=2, norm=norm),  # H//16
            MFunit(channels * 3, channels * 3, g=groups, stride=1, norm=norm),
            MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm),
        )

        # blocks in residual connection
        # TODO: up and down: use conv and deconv
        self.residual11 = MFunit(n + channels, n, g=groups, stride=1, norm=norm)
        self.residual21 = MFunit(channels * 3, channels, stride=1, norm=norm)
        self.residual31 = MFunit(channels * 4, channels * 2, stride=1, norm=norm)
        self.residual41 = MFunit(channels * 2, channels * 2, stride=1, norm=norm)

        self.residual12 = MFunit(n * 2, n, g=groups, stride=1, norm=norm)
        self.residual22 = MFunit(channels * 2 + n, channels, stride=1, norm=norm)
        self.residual32 = MFunit(channels * 5, channels * 2, stride=1, norm=norm)
        self.residual42 = MFunit(channels * 6, channels * 2, stride=1, norm=norm)

        self.connect21 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.connect32 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.connect43 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.connect12 = nn.MaxPool3d(kernel_size=2)
        self.connect23 = nn.MaxPool3d(kernel_size=2)
        self.connect34 = nn.MaxPool3d(kernel_size=2)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        self.decoder_block1 = MFunit(channels * 2 + channels * 2, channels * 2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        self.decoder_block2 = MFunit(channels * 2 + channels, channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2
        self.decoder_block3 = MFunit(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H
        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0, stride=1, bias=False)

        self.softmax = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_block1(x)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        x4 = self.encoder_block4(x3)

        x41 = self.residual41(x4)

        x31 = self.connect43(x41)
        x31 = torch.cat([x3, x31], dim=1)
        x31 = self.residual31(x31)

        x21 = self.connect32(x31)
        x21 = torch.cat([x2, x21], dim=1)
        x21 = self.residual21(x21)

        x11 = self.connect21(x21)
        x11 = torch.cat([x1, x11], dim=1)
        x11 = self.residual11(x11)

        x12 = torch.cat([x1, x11], dim=1)
        x12 = self.residual12(x12)

        x22 = self.connect12(x12)
        x22 = torch.cat([x2, x21, x22], dim=1)
        x22 = self.residual22(x22)

        x32 = self.connect23(x22)
        x32 = torch.cat([x3, x31, x32], dim=1)
        x32 = self.residual32(x32)

        x42 = self.connect34(x32)
        x42 = torch.cat([x4, x41, x42], dim=1)
        x42 = self.residual42(x42)


        # decoder
        y1 = self.upsample1(x42)
        y1 = torch.cat([x32, y1], dim=1)
        y1 = self.decoder_block1(y1)

        y2 = self.upsample2(y1)  # H//4
        y2 = torch.cat([x22, y2], dim=1)
        y2 = self.decoder_block2(y2)

        y3 = self.upsample3(y2)  # H//2
        y3 = torch.cat([x12, y3], dim=1)
        y3 = self.decoder_block3(y3)
        y4 = self.upsample4(y3)
        y4 = self.seg(y4)
        if hasattr(self, 'softmax'):
            y4 = self.softmax(y4)
        return y4



