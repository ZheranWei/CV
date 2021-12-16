from functools import partial
from typing import Sequence
import torch
from torch import nn

from utils.utils_m import ConvNormActivation, SqueezeExcitation as SElayer
from utils.utils_m import _make_divisible

model_path = {
    "mobilenet_v3_large": "weights/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "weights/mobilenet_v3_small-047dcff4.pth",
}


class InvertedResidualConfig:
    def __init__(self, input_channels, kernel, expanded_channels, out_channels, use_se, activation, stride, dilation):
        self.input_channels = input_channels
        self.kernel = kernel
        self.expanded_channels = expanded_channels
        self.out_channels = out_channels
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation


class InvertedResidual(nn.Module):
    def __init__(self, cnf, norm_layer, se_layer=partial(SElayer, scale_activation=nn.Sigmoid)):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU6

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1, norm_layer=norm_layer,
                                   activation_layer=activation_layer)
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel, stride=stride,
                               dilation=cnf.dilation, groups=cnf.expanded_channels, norm_layer=norm_layer,
                               activation_layer=activation_layer,
                               )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(self, inverted_residual_setting, in_channels, last_channel, num_classes=1000, block=None,
                 norm_layer=None,
                 dropout=0.2):
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers = []

        #first layer
        input_channels = in_channels
        output_channels = inverted_residual_setting[0].out_channels
        layers.append(

            ConvNormActivation(input_channels, output_channels, kernel_size=3, stride=1, norm_layer=norm_layer,
                               activation_layer=nn.ReLU6,
                               )

        )

        #inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        #last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1, stride=2,
                               norm_layer=norm_layer,
                               activation_layer=nn.ReLU6,
                               )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)

        if hasattr(self, "avgpool") and hasattr(self, "classifier"):
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

            x = self.classifier(x)

        return x


def _mobilenet_v3_conf(arch):
    bneck_conf = partial(InvertedResidualConfig)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 64, 24, False, "RE", 1, 1),
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C2
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160, True, "HS", 2, 1),  # C4
            bneck_conf(160, 5, 960, 160, True, "HS", 1, 1),
            bneck_conf(160, 5, 960, 160, True, "HS", 1, 1),
        ]
        last_channel = 1280  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 1, 1),
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 240, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "RE", 2, 1),  # C3
            bneck_conf(48, 5, 144, 48, True, "RE", 1, 1),
            bneck_conf(48, 5, 288, 96, True, "RE", 2, 1),  # C4
            bneck_conf(96, 5, 576, 96, True, "RE", 1, 1),
            bneck_conf(96, 5, 576, 96, True, "RE", 1, 1),
        ]
        last_channel = 1024  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def _mobilenet_v3(arch, inverted_residual_setting, in_channels, last_channel, pretrained, num_classes=1000):
    model = MobileNetV3(inverted_residual_setting, in_channels, last_channel, num_classes)
    if pretrained:
        print("pretrained-%s:" % arch, model_path[arch])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path[arch], map_location=device)
        model.load_state_dict(state_dict)

    return model


def mobilenet_v3_large(pretrained=False, in_channels=3, num_classes=1000):
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large")
    model = _mobilenet_v3("mobilenet_v3_large", inverted_residual_setting, in_channels, last_channel, pretrained,
                          num_classes)
    del model.avgpool
    del model.classifier

    return model


def mobilenet_v3_small(pretrained=False, in_channels=3, num_classes=1000):
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small")
    model = _mobilenet_v3("mobilenet_v3_small", inverted_residual_setting, in_channels, last_channel, pretrained,
                          num_classes)
    del model.avgpool
    del model.classifier

    return model
