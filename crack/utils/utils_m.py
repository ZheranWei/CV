import torch.nn as nn


#---------------------------------------------------#
#   返回一个被min_value整除的值
#---------------------------------------------------#
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#---------------------------------------------------#
#   mobilenetv3中自定义卷积模块
#---------------------------------------------------#
class ConvNormActivation(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, dilation=1, inplace=True, bias=None):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups,
                            bias=bias
                            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels

#---------------------------------------------------#
#   mobilenetv3中SE模块
#---------------------------------------------------#
class SqueezeExcitation(nn.Module):

    def __init__(self, input_channels, squeeze_channels, activation=nn.ReLU, scale_activation=nn.Sigmoid):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input


