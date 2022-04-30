import torch
from torch import nn

from src.model.unet_model import UNet


class UpsampleTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.ups = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

    def forward(self, x):
        return self.ups(x)

def test_upsample():
    model = UpsampleTest()
    x = torch.cat((torch.ones(10, 10, 5, 10), torch.zeros(10, 10, 5, 10)), dim=2)
    y = model(x)
    print(y.shape)

def test_unet_model():
    model = UNet(4, 91, bilinear=True)
    X = torch.rand(1, 4, 500, 500)
    Y = model(X)
    print(Y.shape)


if __name__ == '__main__':
    # test_upsample()
    test_unet_model()
