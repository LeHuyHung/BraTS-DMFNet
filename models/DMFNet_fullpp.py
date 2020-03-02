"""
This module reimplement the architecture of UNet++, but instead of using normal convolution layer in the middle
likes in Unet++, we replace it by using MFUnit/DMFUnit
"""

import torch
from torch import nn

try:
    from .sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass

from .DMFNet_16x import normalization, Conv3d_Block, DilatedConv3DBlock, MFunit, DMFUnit


class Up(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2, groups=16, norm='bn'):
        super(Up, self).__init__()
        self.conv = MFunit(n_concat * out_size, out_size, g=groups, stride=1, norm=norm)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                nn.Conv3d(in_size, out_size, 1))

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class DMFNet_fullpp(nn.Module):
    def __init__(self, c=4, n=32, channels=128, groups=16, norm='bn', num_classes=4):
        super(DMFNet_fullpp, self).__init__()

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
        self.up_concat01 = Up(channels, n, is_deconv=False, n_concat=2)
        self.up_concat11 = Up(2 * channels, channels, is_deconv=False, n_concat=2)
        self.up_concat21 = Up(2 * channels, 2 * channels, is_deconv=False, n_concat=2)

        self.up_concat02 = Up(channels, n, is_deconv=False, n_concat=3)
        self.up_concat12 = Up(2 * channels, channels, is_deconv=False, n_concat=3)

        self.up_concat03 = Up(channels, n, is_deconv=False, n_concat=4)

        self.upsample_final_1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upsample_final_2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.upsample_final_3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.final_1 = nn.Conv3d(n, num_classes, 1)
        self.final_2 = nn.Conv3d(n, num_classes, 1)
        self.final_3 = nn.Conv3d(n, num_classes, 1)

        # self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        # self.decoder_block1 = MFunit(channels * 2 + channels * 2, channels * 2, g=groups, stride=1, norm=norm)
        #
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        # self.decoder_block2 = MFunit(channels * 2 + channels, channels, g=groups, stride=1, norm=norm)
        #
        # self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2
        # self.decoder_block3 = MFunit(channels + n, n, g=groups, stride=1, norm=norm)
        # self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H
        # self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0, stride=1, bias=False)

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

        # decoder
        x01 = self.up_concat01(x2, x1)
        x11 = self.up_concat11(x3, x2)
        x21 = self.up_concat21(x4, x3)

        x02 = self.up_concat02(x11, x1, x01)
        x12 = self.up_concat12(x21, x2, x11)

        x03 = self.up_concat03(x12, x1, x01, x02)

        final_1 = self.final_1(self.upsample_final_1(x01))
        final_2 = self.final_2(self.upsample_final_2(x02))
        final_3 = self.final_3(self.upsample_final_3(x03))

        final = (final_1 + final_2 + final_3) / 3


        # y1= self.upsample1(x4)
        # y1 = torch.cat([x3, y1], dim=1)
        # y1 = self.decoder_block1(y1)
        #
        # y2 = self.upsample2(y1)  # H//4
        # y2 = torch.cat([x2, y2], dim=1)
        # y2 = self.decoder_block2(y2)
        #
        # y3 = self.upsample3(y2)  # H//2
        # y3 = torch.cat([x1, y3], dim=1)
        # y3 = self.decoder_block3(y3)
        # y4 = self.upsample4(y3)
        # y4 = self.seg(y4)
        if hasattr(self, 'softmax'):
            y4 = self.softmax(final)
        return y4



