# @Filename:    predict.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        5/2/22 12:16 PM


import sys
import numpy as np
import torch
from modules.blind_net_fft import BlindNetFFT
from PIL import Image, ImageDraw, ImageFont
from data.data_loader import CocoDataset
from torch.utils.data import DataLoader
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')
if __name__ == '__main__':
    model_path = sys.argv[1]

    dataset = CocoDataset(annotations='coco2017/annotations/instances_train2017.json', image_root_dir='coco2017', mask_root_dir='cat_id_masked_arrays', train=True, image_size=32, predict=True)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    image = train_loader.dataset[torch.randint(1, 100, (1,))[0]]
    ground_truth = image[1]
    image_path = image[3]
    rgb_image = Image.open(image_path)
    # plt.imshow(rgb_image)
    # plt.show()
    net = BlindNetFFT(image_size=image[0].shape[1])
    net.load_state_dict(torch.load(model_path))
    net.eval()
    result = net(image[0].unsqueeze(0))
    result = torch.argmax(result, dim=1)
    # result = result.reshape(32, 32)
    print(torch.unique(ground_truth.reshape(32 * 32)))
    print(torch.unique(result))
