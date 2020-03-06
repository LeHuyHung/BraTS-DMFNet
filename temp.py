import torch

from models.unetpp.DMFNet_pp import DMFNet_pp
from models.unetpp.DMFNet_ppd import DMFNet_ppd
from models.unetpp.DMFNet_fullpp import DMFNet_fullpp
from models.attention_unet.DMFNet_attention import DMFNet_attention

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device('cuda:0')
    x = torch.rand((1, 4, 128, 128, 128))  # [bsize,channels,Height,Width,Depth]
    # dmf = DMFNet(c=4, groups=16, norm='sync_bn', num_classes=4)
    # dmfpp = DMFNet_pp(c=4, groups=16, norm='sync_bn', num_classes=4)
    # dmfppd = DMFNet_ppd(c=4, groups=16, norm='sync_bn', num_classes=4)
    # dmffull = DMFNet_fullpp(c=4, groups=16, norm='sync_bn', num_classes=4)
    dmfatt = DMFNet_attention(c=4, groups=16, norm='sync_bn', num_classes=4)
    # dmffull.cuda(device)
    # y = dmfatt(x)
    # print(y.shape)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(count_parameters(dmf))
    # print(count_parameters(dmfpp))
    # print(count_parameters(dmfppd))
    # print(count_parameters(dmffull))
    print(count_parameters(dmfatt))

