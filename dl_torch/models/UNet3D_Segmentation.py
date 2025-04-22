import torch
import torch.nn as nn
import torch.nn.functional as F

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

class UNet3D_16EL(nn.Module):
    def __init__(self, in_channels=1, out_channels=7):
        super().__init__()

        # Encoder (16³ -> 8³)
        self.encoder1 = ConvBlock3D(in_channels, 64)
        self.pool1 = nn.MaxPool3d(2)  # -> 64 × 8³

        # Bottleneck
        self.bottleneck = ConvBlock3D(64, 128)  # -> 128 × 8³

        # Decoder
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)  # -> 64 × 16³
        self.decoder1 = ConvBlock3D(128, 64)  # skip connection from encoder1 (64+64)

        # Output segmentation
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)               # -> 64 × 16³
        pooled = self.pool1(enc1)             # -> 64 × 8³

        bottleneck = self.bottleneck(pooled)  # -> 128 × 8³

        up1 = self.upconv1(bottleneck)        # -> 64 × 16³
        dec1 = self.decoder1(torch.cat((up1, enc1), dim=1))  # -> 64 × 16³

        out = self.final_conv(dec1)           # -> 7 × 16³
        return out