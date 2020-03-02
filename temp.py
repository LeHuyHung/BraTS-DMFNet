import os

import torch

from models import DMFNet
from models.DMFNet_pp import DMFNet_pp
from models.DMFNet_fullpp import DMFNet_fullpp

if __name__ == '__main__':


    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device('cuda:0')
    x = torch.rand((1, 4, 128, 128, 128))  # [bsize,channels,Height,Width,Depth]
    # dmf = DMFNet(c=4, groups=16, norm='sync_bn', num_classes=4)
    # dmfpp = DMFNet_pp(c=4, groups=16, norm='sync_bn', num_classes=4)
    dmffull = DMFNet_fullpp(c=4, groups=16, norm='sync_bn', num_classes=4)
    # dmffull.cuda(device)
    # y = dmffull(x)
    # print(y.shape)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(count_parameters(dmf))
    # print(count_parameters(dmfpp))
    print(count_parameters(dmffull))
