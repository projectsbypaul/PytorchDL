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
    def __init__(self, in_channels=1, out_channels=7, apply_softmax = False):
        super().__init__()
        self.apply_softmax = apply_softmax
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

        logits = self.final_conv(dec1)           # -> 7 × 16³
        return F.softmax(logits, dim=1) if self.apply_softmax else logits

class ConvBNReLU3D(nn.Module):
    """Conv3d → BatchNorm3d → LeakyReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm3d(out_ch)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UNet_Hilbig(nn.Module):
    """
    Hiblig U-like 3D UNet (Fig. 7a):
    1×32³ → 64×32³ → 128×16³ → 256×8³ → 128×16³ → 64×32³ → C×32³
    """
    def __init__(self, in_channels=1, out_channels=7, apply_softmax=False):
        super().__init__()
        self.apply_softmax = apply_softmax

        # ---- Encoder ----
        self.enc1a = ConvBNReLU3D(in_channels, 64)
        self.enc1b = ConvBNReLU3D(64, 64)

        self.pool  = nn.MaxPool3d(2) #64x32³ -> 64x16³

        self.enc2a = ConvBNReLU3D(64, 128)
        self.enc2b = ConvBNReLU3D(128, 128)

        # ---- Bottleneck ----
        self.bott1 = ConvBNReLU3D(128, 256)
        self.bott2 = ConvBNReLU3D(256, 256)

        # ---- Decoder ----
        self.up2   = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2a = ConvBNReLU3D(128 + 128, 128)
        self.dec2b = ConvBNReLU3D(128, 128)

        self.up1   = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1a = ConvBNReLU3D(64 + 64, 64)
        self.dec1b = ConvBNReLU3D(64, 64)

        # ---- Output ----
        self.head  = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder stage 1
        e1 = self.enc1a(x)          # 64 × 32³
        e1 = self.enc1b(e1)         # 64 × 32³
        p1 = self.pool(e1)          # 64 × 16³

        # Encoder stage 2
        e2 = self.enc2a(p1)         # 128 × 16³
        e2 = self.enc2b(e2)         # 128 × 16³
        p2 = self.pool(e2)          # 128 × 8³

        # Bottleneck
        b  = self.bott1(p2)         # 256 × 8³
        b  = self.bott2(b)          # 256 × 8³

        # Decoder stage 2
        u2 = self.up2(b)            # 128 × 16³
        if u2.shape[-3:] != e2.shape[-3:]:
            u2 = F.interpolate(u2, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = torch.cat([u2, e2], dim=1)    # 256 × 16³
        d2 = self.dec2a(d2)                # 128 × 16³
        d2 = self.dec2b(d2)                # 128 × 16³

        # Decoder stage 1
        u1 = self.up1(d2)            # 64 × 32³
        if u1.shape[-3:] != e1.shape[-3:]:
            u1 = F.interpolate(u1, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = torch.cat([u1, e1], dim=1)    # 128 × 32³
        d1 = self.dec1a(d1)                # 64 × 32³
        d1 = self.dec1b(d1)                # 64 × 32³

        # Output
        logits = self.head(d1)       # out_channels × 32³
        return F.softmax(logits, dim=1) if self.apply_softmax else logits
