# @Filename:    BlindNetFFT.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/23/22 2:33 AM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BlindNetFFT(nn.Module):

    def __init__(self, input_size=256, model_name='resnet18', scale_factor=90):
        super(BlindNetFFT, self).__init__()
        if model_name == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        if model_name == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        if model_name == 'resnet34':
            resnet = models.resnet34(pretrained=True)
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])
        self.scale_factor = scale_factor
        RESNET_FEATURE_SIZE = 128
        ## Upsampling Network
        self.upsample = nn.Sequential(
            nn.Conv2d(RESNET_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 91, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, input):
        midlevel_features = self.midlevel_resnet(input)
        output = self.upsample(midlevel_features)
        x = F.softmax(output)
        print(x.shape)
        return x