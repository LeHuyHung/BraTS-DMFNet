import torch

from models.unetpp.DMFNet_pp import DMFNet_pp
from models.unetpp.DMFNet_ppd import DMFNet_ppd
from models.unetpp.DMFNet_fullpp import DMFNet_fullpp
from models.attention_unet.DMFNet_attention import DMFNet_attention
from models import DMFNet_csse
from models import DMFNet_pe, DMFNet_multiattention, DMFNet_attention, DMFNet_singleattention, DMFNet_separate_inputs,\
    DMFNet_pp_double, DMFNet_bifpn, DMFNet_multiscale_weight, DMFNet_interconnect_multiscale_weight, BiFPNNet


if __name__ == '__main__':
    x = torch.rand((1, 4, 128, 128, 128))  # [bsize,channels,Height,Width,Depth]
    model = BiFPNNet(n_layers=3, c=4, n=32, groups=16, channels=128, norm='sync_bn', num_classes=4, bifpn_unit='add')
    # y = model(x)
    # print(y.shape)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))

