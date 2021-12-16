import torch
import torch.nn as nn


class Runet(nn.Module):

    def __init__(self, in_features=3, out_features=2, init_features=32):
        super(Runet, self).__init__()
        features = init_features
        self.encode_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=features, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features * 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=features * 4, out_channels=features * 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=features * 4, out_channels=features * 8, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=features * 8, out_channels=features * 8, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_decode_layer = nn.Sequential(
            nn.Conv2d(in_channels=features * 8, out_channels=features * 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 16),
            nn.ReLU(),
            nn.Conv2d(in_channels=features * 16, out_channels=features * 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 16),
            nn.ReLU()
        )
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decode_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=features * 16, out_channels=features * 8, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=features * 8, out_channels=features * 8, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(),
        )
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decode_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=features * 8, out_channels=features * 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=features * 4, out_channels=features * 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decode_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=features * 4, out_channels=features * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=features * 2, out_channels=features * 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decode_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=features * 2, out_channels=features, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU()
        )
        self.out_layer = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=out_features, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        enc1 = self.encode_layer1(x)
        enc2 = self.encode_layer2(self.pool1(enc1))
        enc3 = self.encode_layer3(self.pool2(enc2))
        enc4 = self.encode_layer4(self.pool3(enc3))

        bottleneck = self.encode_decode_layer(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decode_layer4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decode_layer3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decode_layer2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decode_layer1(dec1)

        out = self.out_layer(dec1)

        return out


def unet():
    return Runet(in_features=3, out_features=2, init_features=32)
