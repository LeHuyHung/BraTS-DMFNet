"""
This model adds MFUnit into each Residual Path, to make the gradient easier in learning. (idea from Unet++ paper)
"""

import torch
from torch import nn
from torch.nn.functional import interpolate

try:
    from models.sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass

from models.DMFNet_16x import normalization, Conv3d_Block, DilatedConv3DBlock, MFunit, DMFUnit


class DMFNet_multiscale_weight(nn.Module):
    def __init__(self, c=4, n=32, channels=128, groups=16, norm='bn', num_classes=4):
        super(DMFNet_multiscale_weight, self).__init__()

        # Entry flow
        self.encoder_block1 = nn.Conv3d(c, n, kernel_size=3, padding=1, stride=2, bias=False)  # H//2
        self.conv = nn.Conv3d(c, groups, kernel_size=3, padding=1, stride=1, bias=False)
        self.encoder_block2 = nn.Sequential(
            DMFUnit(n + groups, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block3 = nn.Sequential(
            DMFUnit(channels + groups, channels * 2, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block4 = nn.Sequential(  # H//8,channels*4
            MFunit(channels * 2 + groups, channels * 3, g=groups, stride=2, norm=norm),  # H//16
            MFunit(channels * 3, channels * 3, g=groups, stride=1, norm=norm),
            MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm),
        )

        # blocks in residual connection
        self.residual1 = MFunit(n, n, g=groups, stride=1, norm=norm)
        self.residual2 = MFunit(channels, channels, stride=1, norm=norm)
        self.residual3 = MFunit(channels * 2, channels * 2, stride=1, norm=norm)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        self.decoder_block1 = MFunit(channels * 2 + channels * 2 + groups, channels * 2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        self.decoder_block2 = MFunit(channels * 2 + channels + groups, channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2
        self.decoder_block3 = MFunit(channels + n + groups, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H
        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0, stride=1, bias=False)

        self.softmax = nn.Softmax(dim=1)

        # weight for inputs
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.w01 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w02 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w04 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w08 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.torch.nn.init.kaiming_normal_(m.weight)  #
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, SynchronizedBatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 4 inputs: /1, /2, /4, /8
        x01 = x * self.w01
        x02 = self.conv(interpolate(x, scale_factor=1/2)) * self.w02
        x04 = self.conv(interpolate(x, scale_factor=1/4)) * self.w04
        x08 = self.conv(interpolate(x, scale_factor=1/8)) * self.w08

        # Encoder
        x1 = self.encoder_block1(x01)
        x1 = torch.cat((x1, x02), dim=1)
        x2 = self.encoder_block2(x1)
        x2 = torch.cat((x2, x04), dim=1)
        x3 = self.encoder_block3(x2)
        x3 = torch.cat((x3, x08), dim=1)
        x4 = self.encoder_block4(x3)

        # decoder
        y1 = self.upsample1(x4)
        y1 = torch.cat([x3, y1], dim=1)
        y1 = self.decoder_block1(y1)

        y2 = self.upsample2(y1)  # H//4
        y2 = torch.cat([x2, y2], dim=1)
        y2 = self.decoder_block2(y2)

        y3 = self.upsample3(y2)  # H//2
        y3 = torch.cat([x1, y3], dim=1)
        y3 = self.decoder_block3(y3)
        y4 = self.upsample4(y3)
        y4 = self.seg(y4)
        if hasattr(self, 'softmax'):
            y4 = self.softmax(y4)
        return y4
