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

# TODO: not done
class DMFNet_fc(nn.Module):
    def __init__(self, c=4, n=32, channels=128, groups=16, norm='bn', num_classes=4):
        super(DMFNet_fc, self).__init__()

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

        # # blocks in residual connection
        # self.residual1 = MFunit(n, n, g=groups, stride=1, norm=norm)
        # self.residual2 = MFunit(channels, channels, stride=1, norm=norm)
        # self.residual3 = MFunit(channels * 2, channels * 2, stride=1, norm=norm)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        self.decoder_block1 = MFunit(channels * 2 + channels * 2, channels * 2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        self.decoder_block2 = MFunit(channels * 2 + channels, channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2
        self.decoder_block3 = MFunit(channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H
        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0, stride=1, bias=False)

        self.softmax = nn.Softmax(dim=1)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.w11 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w12 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w13 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w14 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)

        self.w21 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w22 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w23 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w24 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)

        self.w31 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w32 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w33 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)
        self.w34 = torch.tensor(1.0, dtype=torch.float, requires_grad=True).to(device)

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
