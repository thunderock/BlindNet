import torch
from torch import nn

from src.model.unet_model import DoubleConv, Down, Up, OutConv


class SingleMaskCnn(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SingleMaskCnn, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 512)
        self.down6 = Down(512, 512)
        self.fc1 = nn.Linear(in_features=512 * 6 * 6, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=91)

        self.seq = torch.nn.Sequential(
            self.inc,
            self.down1,
            self.down2,
            self.down3,
            self.down4,
            self.down5,
            self.down6,
        )
        self.lins = torch.nn.Sequential(
            self.fc1,
            self.fc2
        )

    def forward(self, x):
        x5 = self.seq(x)
        return self.lins(x5.view(x5.shape[0], -1))
