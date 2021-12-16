import torch
import torch.nn as nn
from torchsummary import summary

from nets.vgg import VGG16
from nets.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from utils.utils_m import ConvNormActivation
from nets.unet import unet


def get_features(channel=3, w=480, h=320, backbone="Mobilenet_v3_small"):
    if backbone == "Vgg16":
        model = VGG16()
    elif backbone == "Mobilenet_v3_small":
        model = mobilenet_v3_small()
    else:
        model = mobilenet_v3_large()
    sx = torch.randn(2, channel, w, h)

    outlist = []
    downlist = []
    features = {}
    downfeatures = {}
    outfeatures = {}
    model = model.features
    iw, ow, x, scale, c, dscale, count = w, 512, 512, 0.5, 3, 2, 0
    for i, f in enumerate(model):
        if i == 0:
            iw = sx.shape[-1]
            c = sx.shape[1]
            x = f(sx)
            ow = f(sx).shape[-1]
            scale = float(ow / iw)
            dscale = sx.shape[-1] / ow
        else:
            iw = x.shape[-1]
            c = x.shape[1]
            ow = f(x).shape[-1]
            scale = float(ow / iw)
            dscale = sx.shape[-1] / ow
            x = f(x)
        features["features%d" % i] = [[int(dscale), scale, iw, ow], [c, x.shape[1]]]
        if scale != 1.0:
            downfeatures["features%d" % i] = [[int(dscale), scale, iw, ow], [c, x.shape[1]]]
            downlist.append(i)
            count += 1
            outlist.append(i - 1)

    lenfeatures = len(features)
    outlist[-1] = lenfeatures - 2

    for i, o in enumerate(outlist):
        item = "features%d" % o
        outfeatures[item] = features[item]

    index = downlist[:-1]
    index.append(lenfeatures - 1)

    # print("downlist:", downlist, "outlist:", outlist)
    # print("downfeatures:\n", downfeatures)
    # print("outfeatures:\n", outfeatures)
    return outlist, index, features, lenfeatures


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = ConvNormActivation(in_size, out_size)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, pretrained=False, backbone="Mobilenet_v3_small"):
        super(Unet, self).__init__()
        self.flag = 0
        if backbone == "Vgg16":
            self.backbone = VGG16(pretrained=pretrained, in_channels=in_channels)
        elif backbone == "Mobilenet_v3_small":
            self.backbone = mobilenet_v3_small(pretrained=pretrained, in_channels=in_channels)
        elif backbone == "Mobilenet_v3_large":
            self.backbone = mobilenet_v3_large(pretrained=pretrained, in_channels=in_channels)
        elif backbone == "Unet":
            if pretrained==True:
                print("This backbone have no pretrained weights")
            self.flag = 1
            self.backbone = unet()
        else:
            raise ValueError("No such backbone:%s" % backbone)

        """
        Mobilenet_v3_small.featuresi:[[ndownsample, iw/ow, iw, ow], [ic, oc]]
        # downfeatures:所有降采样层
        # {'features2': [[2, 0.5, 320, 160], [16, 24]], 'features5': [[4, 0.5, 160, 80], [40, 40]],
        #  'features7': [[8, 0.5, 80, 40], [40, 48]], 'features9': [[16, 0.5, 40, 20], [48, 96]],
        #  'features12': [[32, 0.5, 20, 10], [96, 576]]}
        # outfeatures:根据降采样层取前四次降采样后的结果(详见函数get_features)，方便后期特征融合
        # {'features1': [[1, 1.0, 320, 320], [16, 16]], 'features4': [[2, 1.0, 160, 160], [24, 40]],
        #  'features6': [[4, 1.0, 80, 80], [40, 40]], 'features8': [[8, 1.0, 40, 40], [48, 48]],
        #  'features11': [[16, 1.0, 20, 20], [96, 96]]}
        """

        if self.flag != 1:
            outlist, self.index, features, lenfeatures = get_features(backbone=backbone)
            in_filters = []
            out_filters = []
            for i, o in enumerate(outlist):
                if i != len(outlist) - 1: out_filters.append(features["features%d" % o][1][1])
            for i, o in enumerate(out_filters):
                if i != len(out_filters) - 1: in_filters.append(out_filters[i] + out_filters[i + 1])
            in_filters.append(out_filters[-1] + features["features%d" % outlist[-1]][1][1])

            self.up_concat4 = unetUp(in_filters[3], out_filters[3])
            self.up_concat3 = unetUp(in_filters[2], out_filters[2])
            self.up_concat2 = unetUp(in_filters[1], out_filters[1])
            self.up_concat1 = unetUp(in_filters[0], out_filters[0])

            self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        if self.flag == 1:
            return self.backbone(inputs)

        feat1 = self.backbone.features[:self.index[0]](inputs)
        feat2 = self.backbone.features[self.index[0]:self.index[1]](feat1)
        feat3 = self.backbone.features[self.index[1]:self.index[2]](feat2)
        feat4 = self.backbone.features[self.index[2]:self.index[3]](feat3)
        feat5 = self.backbone.features[self.index[3]:self.index[4]](feat4)
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        final = self.final(up1)

        return final


if __name__ == '__main__':
    x = torch.randn(1, 3, 320, 480).cuda()
    net = Unet()
    summary(net.cuda(), (3, 320, 480))
    print(net(x).shape)
