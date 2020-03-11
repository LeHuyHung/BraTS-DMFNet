"""
This model adds MFUnit into each Residual Path, to make the gradient easier in learning. (idea from Unet++ paper)
"""

import torch
from torch import nn
import numpy as np

try:
    from models.sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass

from models.DMFNet_16x import MFunit, DMFUnit


def croppCenter(tensorToCrop, finalShape):
    org_shape = tensorToCrop.shape

    diff = np.zeros(3)
    diff[0] = org_shape[2] - finalShape[2]
    diff[1] = org_shape[3] - finalShape[3]
    diff[1] = org_shape[4] - finalShape[4]

    croppBorders = np.zeros(3, dtype=int)
    croppBorders[0] = int(diff[0] / 2)
    croppBorders[1] = int(diff[1] / 2)
    croppBorders[2] = int(diff[2] / 2)

    return tensorToCrop[:,
           :,
           croppBorders[0]:org_shape[2] - croppBorders[0],
           croppBorders[1]:org_shape[3] - croppBorders[1],
           croppBorders[2]:org_shape[4] - croppBorders[2]]


# TODO: in encoder, just use Conv2D
class DMFNet_separate_inputs(nn.Module):
    def __init__(self, c=4, n=32, channels=128, groups=16, norm='bn', num_classes=4):
        super(DMFNet_separate_inputs, self).__init__()

        # Entry flow (4 blocks for 4 modals)
        # Modal 1
        self.encoder_block10 = nn.Conv3d(1, n, kernel_size=3, padding=1, stride=2, bias=False)  # H//2
        self.encoder_block20 = nn.Sequential(
            DMFUnit(n * 4, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block30 = nn.Sequential(
            DMFUnit(channels * 4, channels * 2, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block40 = nn.Sequential(  # H//8,channels*4
            MFunit(channels * 8, channels * 3, g=groups, stride=2, norm=norm),  # H//16
            MFunit(channels * 3, channels * 3, g=groups, stride=1, norm=norm),
            MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm),
        )

        # Modal 2
        self.encoder_block11 = nn.Conv3d(1, n, kernel_size=3, padding=1, stride=2, bias=False)  # H//2
        self.encoder_block21 = nn.Sequential(
            DMFUnit(n * 4, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block31 = nn.Sequential(
            DMFUnit(channels * 4, channels * 2, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block41 = nn.Sequential(  # H//8,channels*4
            MFunit(channels * 8, channels * 3, g=groups, stride=2, norm=norm),  # H//16
            MFunit(channels * 3, channels * 3, g=groups, stride=1, norm=norm),
            MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm),
        )

        # Modal 3
        self.encoder_block12 = nn.Conv3d(1, n, kernel_size=3, padding=1, stride=2, bias=False)  # H//2
        self.encoder_block22 = nn.Sequential(
            DMFUnit(n * 4, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block32 = nn.Sequential(
            DMFUnit(channels * 4, channels * 2, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block42 = nn.Sequential(  # H//8,channels*4
            MFunit(channels * 8, channels * 3, g=groups, stride=2, norm=norm),  # H//16
            MFunit(channels * 3, channels * 3, g=groups, stride=1, norm=norm),
            MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm),
        )

        # Modal 4
        self.encoder_block13 = nn.Conv3d(1, n, kernel_size=3, padding=1, stride=2, bias=False)  # H//2
        self.encoder_block23 = nn.Sequential(
            DMFUnit(n * 4, channels, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//4 down
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels, channels, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block33 = nn.Sequential(
            DMFUnit(channels * 4, channels * 2, g=groups, stride=2, norm=norm, dilation=[1, 2, 3]),  # H//8
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3]),  # Dilated Conv 3
            DMFUnit(channels * 2, channels * 2, g=groups, stride=1, norm=norm, dilation=[1, 2, 3])
        )

        self.encoder_block43 = nn.Sequential(  # H//8,channels*4
            MFunit(channels * 8, channels * 3, g=groups, stride=2, norm=norm),  # H//16
            MFunit(channels * 3, channels * 3, g=groups, stride=1, norm=norm),
            MFunit(channels * 3, channels * 2, g=groups, stride=1, norm=norm),
        )

        # bridge
        self.bridge = MFunit(channels * 8, channels * 2, g=groups, stride=1, norm=norm)

        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        self.decoder_block1 = MFunit(channels * 2, channels * 2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        self.decoder_block2 = MFunit(channels * 2, channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2
        self.decoder_block3 = MFunit(channels, n, g=groups, stride=1, norm=norm)

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
        # Separate inputs
        i0 = x[:, 0:1, :, :, :]
        i1 = x[:, 1:2, :, :, :]
        i2 = x[:, 2:3, :, :, :]
        i3 = x[:, 3:4, :, :, :]

        # -----  First Level --------
        down_1_0 = self.encoder_block10(i0)
        down_1_1 = self.encoder_block11(i1)
        down_1_2 = self.encoder_block12(i2)
        down_1_3 = self.encoder_block13(i3)

        # -----  Second Level --------
        # input_2nd = torch.cat((down_1_0,down_1_1,down_1_2,down_1_3),dim=1)
        input_2nd_0 = torch.cat((down_1_0, down_1_1, down_1_2, down_1_3), dim=1)

        input_2nd_1 = torch.cat((down_1_1, down_1_2, down_1_3, down_1_0), dim=1)

        input_2nd_2 = torch.cat((down_1_2, down_1_3, down_1_0, down_1_1), dim=1)

        input_2nd_3 = torch.cat((down_1_3, down_1_0, down_1_1, down_1_2), dim=1)

        down_2_0 = self.encoder_block20(input_2nd_0)
        down_2_1 = self.encoder_block21(input_2nd_1)
        down_2_2 = self.encoder_block22(input_2nd_2)
        down_2_3 = self.encoder_block23(input_2nd_3)

        # Level 3
        input_3rd_0 = torch.cat((down_2_0, down_2_1, down_2_2, down_2_3), dim=1)
        # cropped_2 = croppCenter(input_2nd_0, input_3rd_0.shape)
        # input_3rd_0 = torch.cat((input_3rd_0, cropped_2), dim=1)

        input_3rd_1 = torch.cat((down_2_1, down_2_2, down_2_3, down_2_0), dim=1)
        # input_3rd_1 = torch.cat((input_3rd_1, croppCenter(input_2nd_1, input_3rd_1.shape)), dim=1)

        input_3rd_2 = torch.cat((down_2_2, down_2_3, down_2_0, down_2_1), dim=1)
        # input_3rd_2 = torch.cat((input_3rd_2, croppCenter(input_2nd_2, input_3rd_2.shape)), dim=1)

        input_3rd_3 = torch.cat((down_2_3, down_2_0, down_2_1, down_2_2), dim=1)
        # input_3rd_3 = torch.cat((input_3rd_3, croppCenter(input_2nd_3, input_3rd_3.shape)), dim=1)

        down_3_0 = self.encoder_block30(input_3rd_0)
        down_3_1 = self.encoder_block31(input_3rd_1)
        down_3_2 = self.encoder_block32(input_3rd_2)
        down_3_3 = self.encoder_block33(input_3rd_3)

        # -----  Fourth Level --------

        # Max-pool
        input_4th_0 = torch.cat((down_3_0, down_3_1, down_3_2, down_3_3), dim=1)
        # input_4th_0 = torch.cat((input_4th_0,croppCenter(input_3rd_0, input_4th_0.shape)), dim=1)

        input_4th_1 = torch.cat((down_3_1, down_3_2, down_3_3, down_3_0), dim=1)
        # input_4th_1 = torch.cat((input_4th_1,croppCenter(input_3rd_1, input_4th_1.shape)), dim=1)

        input_4th_2 = torch.cat((down_3_2, down_3_3, down_3_0, down_3_1), dim=1)
        # input_4th_2 = torch.cat((input_4th_2,croppCenter(input_3rd_2, input_4th_2.shape)), dim=1)

        input_4th_3 = torch.cat((down_3_3, down_3_0, down_3_1, down_3_2), dim=1)
        # input_4th_3 = torch.cat((input_4th_3,croppCenter(input_3rd_3, input_4th_3.shape)), dim=1)

        down_4_0 = self.encoder_block40(input_4th_0)
        down_4_1 = self.encoder_block41(input_4th_1)
        down_4_2 = self.encoder_block42(input_4th_2)
        down_4_3 = self.encoder_block43(input_4th_3)

        inputBridge = torch.cat((down_4_0, down_4_1, down_4_2, down_4_3), dim=1)
        bridge = self.bridge(inputBridge)

        # Decoder
        skip_1 = (bridge + down_4_0 + down_4_1 + down_4_2 + down_4_3) / 5  # Residual connection
        up_1 = self.upsample1(skip_1)
        deconv_2 = self.decoder_block1(up_1)
        skip_2 = (deconv_2 + down_3_0 + down_3_1 + down_3_2 + down_3_3) / 5  # Residual connection
        up_2 = self.upsample2(skip_2)
        deconv_3 = self.decoder_block2(up_2)
        skip_3 = (deconv_3 + down_2_0 + down_2_1 + down_2_2 + down_2_3) / 5  # Residual connection
        up_3 = self.upsample3(skip_3)
        deconv_4 = self.decoder_block3(up_3)
        skip_4 = (deconv_4 + down_1_0 + down_1_1 + down_1_2 + down_1_3) / 5  # Residual connection
        up_4 = self.upsample4(skip_4)

        # # Encoder
        # x1 = self.encoder_block1(x)
        # x2 = self.encoder_block2(x1)
        # x3 = self.encoder_block3(x2)
        # x4 = self.encoder_block4(x3)
        #
        # # decoder
        # y1 = self.upsample1(x4)
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
        y4 = self.seg(up_4)
        if hasattr(self, 'softmax'):
            y4 = self.softmax(y4)
        return y4
