# @Filename:    data_loader.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/2/22 10:24 PM

import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import skimage.io as io
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from pycocotools.coco import COCO, maskUtils
import matplotlib
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
matplotlib.use('TkAgg')


class CocoDataset(Dataset):
    @staticmethod
    def get_img_path(img_name, dir, root_dir='../coco2017/'):
        x = format(img_name, '012d')
        # print(x, type(x))
        return os.path.join(root_dir, dir, '{}.jpg'.format(x))

    def __init__(self, annotations, root_dir, dir):
        super().__init__()
        self.coco = COCO(annotations)
        self.img_ids = self.coco.getImgIds()
        self.dir = dir
        self.root_dir = root_dir
        self.resize = T.Resize(size=(256, 256))
        self.transform = T.Compose([T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = T.Compose([T.ToTensor()])

    def write_masked_array(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        img = Image.open(CocoDataset.get_img_path(img_id, self.dir, self.root_dir))
        transformed = CocoDataset.get_img_numpy_array(anns, np.array(img), img_id)
        np.save('../coco2017/cat_id_masked_arrays/{}/{}.npy'.format(self.dir, img_id), transformed)
        return True

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def get_img_numpy_array(anns, img, img_id):
        array = np.zeros_like(img, dtype=np.uint8)
        # polygons = []
        # cat = []
        for ann in anns:
            cat_id = ann["category_id"]
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:
                    # polygon
                    for seg in ann['segmentation']:
                        poly = Polygon(np.array(seg).reshape((int(len(seg) / 2), 2)))
                        # polygons.append(poly)
                        # cat.append(cat_id)
                        for i in range(img.shape[0]):
                            # print(i)
                            for j in range(img.shape[1]):
                                if poly.contains_point((i, j)):
                                    array[i, j, :] = cat_id
                else:
                    assert False, "failing {}".format(img_id)
        return array

    def __getitem__(self, idx):
        return self.write_masked_array(self.img_ids[idx])

        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        img = Image.open(self.get_img_path(img_id, self.dir, self.root_dir))
        transformed = self.get_img_numpy_array(anns, np.array(img), img_id)
        return self.transform(self.resize(img)), \
               self.target_transform(self.resize(Image.fromarray(transformed)))




ds = CocoDataset('../coco2017/annotations/instances_val2017.json', '../coco2017', 'val2017')

trainloader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=20)


status_loop = tqdm(trainloader, total=len(trainloader), leave=True)

for i, data in status_loop:
    pass


ds = CocoDataset('../coco2017/annotations/instances_train2017.json', '../coco2017', 'train2017')

trainloader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=20)

status_loop = tqdm(trainloader, total=len(trainloader), leave=True)

for i, data in status_loop:
    pass



