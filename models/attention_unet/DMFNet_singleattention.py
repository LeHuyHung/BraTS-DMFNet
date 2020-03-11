"""
This model adds Attention Gate into each Residual Path, to make the gradient easier in learning. (idea from Unet++ paper)
"""

import torch
from torch import nn
import torch.nn.functional as F

try:
    from models.sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass

from models.DMFNet_16x import normalization, Conv3d_Block, DilatedConv3DBlock, MFunit, DMFUnit
from .grid_attention_layer import init_weights, GridAttentionBlock3D


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]

        if offset % 2 == 0:
            padding = [0, 0, offset // 2, offset // 2, offset // 2, offset // 2]
        else:
            padding = [0, 0, offset // 2, 0, 0, offset // 2]
        if offset > 0:
            outputs1 = F.pad(inputs1, padding)
        else:
            outputs1 = inputs1
            padding = [e * -1 for e in padding]
            outputs2 = F.pad(outputs2, padding)

        return self.conv(torch.cat([outputs1, outputs2], 1))


# TODO: should we use Deep Super Vision
class DMFNet_singleattention(nn.Module):
    def __init__(self, c=4, n=32, channels=128, groups=16, norm='bn', num_classes=4, is_batchnorm=True,
                 nonlocal_mode='concatenation', attention_dsample=(2, 2, 2), is_deconv=False):
        super(DMFNet_singleattention, self).__init__()

        self.is_batchnorm = is_batchnorm
        self.is_deconv = is_deconv
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

        # FOR ATTENTION GATE
        self.up_concat = UnetUp3(channels * 4, channels * 2, self.is_deconv, self.is_batchnorm)

        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.center = UnetConv3(channels * 2, channels * 4, self.is_batchnorm)
        gating_channels = 2 * channels  # TODO
        self.gating = UnetGridGatingSignal3(channels * 4, gating_channels, kernel_size=(1, 1, 1),
                                            is_batchnorm=self.is_batchnorm)

        # attention blocks
        #  in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor
        self.attentionblock1 = MultiAttentionBlock(in_size=n, gate_size=gating_channels,
                                                   inter_size=channels, sub_sample_factor=attention_dsample,
                                                   nonlocal_mode=nonlocal_mode)
        self.attentionblock2 = MultiAttentionBlock(in_size=channels, gate_size=gating_channels,
                                                   inter_size=channels, sub_sample_factor=attention_dsample,
                                                   nonlocal_mode=nonlocal_mode)
        self.attentionblock3 = MultiAttentionBlock(in_size=2 * channels, gate_size=gating_channels,
                                                   inter_size=channels, sub_sample_factor=attention_dsample,
                                                   nonlocal_mode=nonlocal_mode)
        self.attentionblock4 = MultiAttentionBlock(in_size=channels * 2, gate_size=gating_channels,
                                                   inter_size=channels, sub_sample_factor=attention_dsample,
                                                   nonlocal_mode=nonlocal_mode)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//8
        self.decoder_block1 = MFunit(channels * 2 + channels * 2, channels * 2, g=groups, stride=1, norm=norm)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//4
        self.decoder_block2 = MFunit(channels * 2 + channels, 2 * channels, g=groups, stride=1, norm=norm)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H//2
        self.decoder_block3 = MFunit(2 * channels + n, n, g=groups, stride=1, norm=norm)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # H

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=n, out_size=num_classes, scale_factor=2)
        self.dsv3 = UnetDsv3(in_size=channels * 2, out_size=num_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=channels * 2, out_size=num_classes, scale_factor=8)

        self.seg = nn.Conv3d(n, num_classes, kernel_size=1, padding=0, stride=1, bias=False)
        self.final = nn.Conv3d(num_classes * 3, num_classes, 1)

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
        # x1_res = self.residual1(x1)
        x2 = self.encoder_block2(x1)
        # x2_res = self.residual2(x2)
        x3 = self.encoder_block3(x2)
        # x3_res = self.residual3(x3)
        x4 = self.encoder_block4(x3)

        # Gating Signal Generation
        maxpool = self.maxpool(x4)
        center = self.center(maxpool)
        gating = self.gating(center)

        # Attention Mechanism
        g_conv4, att4 = self.attentionblock4(x4, gating)
        y0 = self.up_concat(g_conv4, center)

        g_conv3, att3 = self.attentionblock3(x3, y0)
        y1 = self.upsample2(y0)  # H//4
        y1 = torch.cat([g_conv3, y1], dim=1)
        y1 = self.decoder_block1(y1)

        g_conv2, att2 = self.attentionblock2(x2, y1)
        y2 = self.upsample3(y1)  # H//2
        y2 = torch.cat([g_conv2, y2], dim=1)
        y2 = self.decoder_block2(y2)

        g_conv1, att1 = self.attentionblock1(x1, y2)
        y3 = self.upsample4(y2)
        y3 = torch.cat([g_conv1, y3], dim=1)
        y3 = self.decoder_block3(y3)

        # y4 = self.upsample4(y3)

        # Deep Supervision
        dsv4 = self.dsv4(y3)
        dsv3 = self.dsv3(y2)
        dsv2 = self.dsv2(y1)
        # y4 = self.seg(y4)

        y4 = self.final(torch.cat([dsv2, dsv3, dsv4], dim=1))

        if hasattr(self, 'softmax'):
            y4 = self.softmax(y4)
        return y4


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1
