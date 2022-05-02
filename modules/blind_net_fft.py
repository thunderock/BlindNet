# @Filename:    blind_net_fft.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/23/22 2:33 AM

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BlindNetFFT(nn.Module):

    def __init__(self, model_name='resnet18', scale_factor=90, image_size=84):
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
        self.image_size = image_size
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
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.linear = nn.Sequential(nn.Linear(12288, 1024), nn.ReLU(), nn.Linear(1024, 91), nn.Sigmoid(), )

    def forward(self, input):
        batch_size = input.size(0)
        # using only the first channel of image
        input = input[:, 0, :,:].reshape(batch_size, 1, input.size(2), input.size(3)).float()
        midlevel_features = self.midlevel_resnet(input)
        output = self.upsample(midlevel_features)
        # print(output.size())
        # print(output.size())
        x = self.linear(output)
        # print(output.size())
        # x = F.log_softmax(output.view(batch_size, 91, self.image_size, self.image_size), dim=1)
        # print(x[0, :, 0, 0])
        # output should be batch_size * (256 * 256) * 91 *
        # x = output.view(batch_size * self.image_size * self.image_size, 91)
        return x
        # x = torch.max(x, dim=1)[1]
        # print(x)
        # return F.one_hot(x, 91)