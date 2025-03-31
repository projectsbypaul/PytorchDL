import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized.functional import conv1d


class Encoder(nn.Module):
    def __init__(self, n_feature : int):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.pool_2= nn.MaxPool3d(kernel_size=2, padding=0)

        self.mlp_down = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_feature),
            nn.Softmax(dim=1))


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool_2(F.relu(x))
        x = self.conv2(x)
        x = self.pool_2(F.relu(x))
        x = self.conv3(x)
        x = self.pool_2(F.relu(x))
        x = self.conv4(x)
        x = self.pool_2(F.relu(x))
        x = self.conv5(x)
        x = self.pool_2(F.relu(x))

        x = x.view(x.size(0), -1) #flatten

        x = self.mlp_down(x)

        return x


class Decoder(nn.Module):
    def __init__(self, n_feature: int =16):
        super(Decoder, self).__init__()

        self.mlp_up = nn.Sequential(
            nn.Linear(n_feature, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.conv1 = nn.Conv3d(32, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, padding=1)





    def forward(self, x):
        x = self.mlp_up(x)
        x = x.view(x.size(0), 64, 1, 1, 1)

        x = F.tanh(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.conv5(x)
        x = F.tanh(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.conv4(x)
        x = F.tanh(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.conv3(x)
        x = F.tanh(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.conv2(x)
        x = F.tanh(F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False))
        x = self.conv1(x)
        x = F.tanh(x)
        return x

class Decoder_binary(nn.Module):
    def __init__(self, n_feature: int = 16):
        super(Decoder_binary, self).__init__()

        self.mlp_up = nn.Sequential(
            nn.Linear(n_feature, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.conv1 = nn.Conv3d(32, 1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.mlp_up(x)
        x = x.view(x.size(0), 64, 1, 1, 1)

        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        x = self.conv5(x)
        x = F.interpolate(F.relu(x), scale_factor=2, mode="trilinear", align_corners=False)
        x = self.conv4(x)
        x = F.interpolate(F.relu(x), scale_factor=2, mode="trilinear", align_corners=False)
        x = self.conv3(x)
        x = F.interpolate(F.relu(x), scale_factor=2, mode="trilinear", align_corners=False)
        x = self.conv2(x)
        x = F.interpolate(F.relu(x), scale_factor=2, mode="trilinear", align_corners=False)
        x = self.conv1(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, n_feature : int = 16):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(n_feature)
        self.decoder = Decoder(n_feature)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class Autoencoder_binary(nn.Module):
    def __init__(self, n_feature : int = 16):
        super(Autoencoder_binary, self).__init__()
        self.encoder = Encoder(n_feature)
        self.decoder = Decoder_binary(n_feature)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

def main():
    # Instantiate the model
    autoencoder = Autoencoder_binary()
    # print(autoencoder)

if __name__  =="__main__":
    main()
