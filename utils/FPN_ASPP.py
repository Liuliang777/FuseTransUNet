
#
import  numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class FPN_ASPP(nn.Module):
    def __init__(self):
        super(FPN_ASPP,self).__init__()
        self.inplanes = 64


        self.toplayer = nn.Sequential(nn.Conv2d(256,256,1,1,0),
                                        ASPP(in_channels=256, atrous_rates=[1,2,3]),
                                        ASPP(in_channels=256, atrous_rates=[1,2,3]),
                                        nn.Conv2d(256,256,1,1,0))
        self.toplayer2 = nn.Sequential(nn.Conv2d(128, 256, 1, 1, 0),
                                      ASPP(in_channels=256, atrous_rates=[1, 2, 3]),
                                      ASPP(in_channels=256, atrous_rates=[1, 2, 3]),
                                      nn.Conv2d(256, 256, 1, 1, 0))
        self.adjust1 = nn.Conv2d(512, 128, 1, 1, 0)
        self.adjust2 = nn.Conv2d(192, 64, 1, 1, 0)


    def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return F.interpolate(x,size=(H,W),mode='bilinear',align_corners=False) + y

    def _upsample_cat(self, x, y):
        _,_,H,W = y.shape
        x = F.interpolate(x,size=(H,W),mode='bilinear',align_corners=False)
        return torch.cat((x,y),dim=1)

    def forward(self,x):
        c3,c4,c5 = x

        #自上而下
        p5 = self.toplayer(c5)
        p4 = self.toplayer2(c4)  # 256
        p4 = self._upsample_cat(p5, p4)
        p4 = self.adjust1(p4)   #128
        p3 = self._upsample_cat(p4, c3)   # 256
        p3 = self.adjust2(p3)  # 256


        return p3

if __name__ == "__main__":
    fpn = FPN_ASPP()
    fpn = fpn.cuda()
    #c2 = torch.randn(1,64,512,512).cuda()
    c3 = torch.randn(1,64,64,64).cuda()
    c4 = torch.randn(1,128, 32, 32).cuda()
    c5 = torch.randn(1,256, 16, 16).cuda()
    x = [c3,c4,c5]
    y = fpn(x)
    print(y.shape)

