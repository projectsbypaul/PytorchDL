import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class AutoEncoder3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.encoder1 = ConvBlock3D(in_channels, 32)  # was 64
        self.pool1 = nn.MaxPool3d(2)  # 16³ -> 8³

        # Bottleneck
        self.bottleneck = ConvBlock3D(32, 64)  # was 128

        # Decoder
        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)  # 8³ -> 16³
        self.decoder1 = ConvBlock3D(64, 32)  # skip connection (32+32)

        # Output (reconstruction)
        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)
        self.final_activation = nn.Tanh()  # Because your SDF values are in [-1, 1]

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pooled = self.pool1(enc1)

        # Bottleneck
        bottleneck = self.bottleneck(pooled)

        # Decoder
        up1 = self.upconv1(bottleneck)
        dec1 = self.decoder1(torch.cat((up1, enc1), dim=1))

        # Output
        out = self.final_conv(dec1)
        out = self.final_activation(out)
        return out