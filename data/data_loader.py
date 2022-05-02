# @Filename:    data_loader.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/2/22 10:24 PM

import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from pycocotools.coco import COCO
from matplotlib.patches import Polygon
from PIL import Image
from tqdm import tqdm


class CocoDataset(Dataset):
    @staticmethod
    def get_img_path(img_name, root_dir, dir):
        x = format(img_name, '012d')
        # print(x, type(x))
        return os.path.join(root_dir, dir, '{}.jpg'.format(x))


    def __init__(self, annotations, image_root_dir, mask_root_dir, train, image_size=84, predict=False):
        super().__init__()
        self.coco = COCO(annotations)
        self.img_ids = self.coco.getImgIds()
        self.mask_root_dir = mask_root_dir
        self.image_root_dir = image_root_dir
        self.dir = 'train2017' if train else 'val2017'
        self.resize = T.Resize(size=(image_size, image_size))
        self.transform = T.Compose([T.ToTensor()]) if predict else T.Compose([T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = T.Compose([T.ToTensor()])
        self.train = train
        self.image_size = image_size
        self.predict = predict

    def write_masked_array(self, img_id):
        transformed_file = '../coco2017/cat_id_masked_arrays/{}/{}.npy'.format(self.dir, img_id)
        img_path = CocoDataset.get_img_path(img_id, self.dir, self.image_root_dir)

        if os.path.exists(transformed_file):
            try:
                os.remove(img_path)
            except:
                pass
            return True

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        img = Image.open(img_path)
        transformed = CocoDataset.get_img_numpy_array(anns, np.array(img), img_id)
        os.remove(img_path)
        np.save(transformed_file, transformed)
        return True

    def __len__(self):
        return len(self.img_ids)

    @staticmethod
    def get_img_numpy_array(anns, img, img_id):
        shape = img.shape
        array = np.zeros(shape[:2], dtype=np.uint8)
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
                                    array[i, j] = cat_id
                else:
                    assert False, "failing {}".format(img_id)
        return array



    def check_and_correct_image(self, X):

        l, w = X.shape[:2]
        if X.shape != (l, w, 3):
            # only one channel
            pad = np.zeros((l, w, 1))
            if len(X.shape) == 2:
                X = np.expand_dims(X, axis=-1)
                X = np.dstack((X, pad))
            if X.shape != (l, w, 3):
                X = np.dstack((X, pad))
        return X

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        transformed_file = '{}/{}/{}.npy'.format(self.mask_root_dir, self.dir, img_id)
        img_path = self.get_img_path(img_id, self.image_root_dir, self.dir)

        img = np.array(Image.open(img_path))
        X = self.transform(self.check_and_correct_image(img))
        y = np.load(transformed_file)
        y = torch.tensor(y)

        X = self.resize(X)
        y = self.resize(y.unsqueeze(0))
        i, j, h, w = T.RandomCrop.get_params(X, output_size=(self.image_size, self.image_size))
        X = TF.crop(X, i, j, h, w)
        # channel first

        y = TF.crop(y, i, j, h, w).squeeze(0)

        if np.random.rand() > 0.5:
            X = TF.hflip(X)
            y = TF.hflip(y)

        # get all the masks
        distinct_cats = torch.unique(y)
        # assert X.shape == (3, 256, 256) and y.shape == (256, 256), "Image shape is not correct {}: img shape: {} target shape: {}".format(img_id, X.shape, y.shape)

        # assert each value between 1 and 90
        # assert (torch.max(distinct_cats) <= 90 and torch.min(distinct_cats) >= 0), "Categories: {}".format(distinct_cats)
        # # assert size of distint_cats is at least 2
        # assert len(distinct_cats) >= 1
        random_cat = distinct_cats[torch.randint(0, len(distinct_cats), (1,))]
        # y_hat = torch.clone(y)
        # y[y == random_cat] = -1
        # X is the image, y is the mask, random_cat is a random cat id in the mask
        if self.predict:
            return X, y, random_cat, img_path
        return X, y, random_cat



# ds = CocoDataset(annotations='../coco2017/annotations/instances_val2017.json',
#                  image_root_dir='../coco2017', mask_root_dir='../cat_id_masked_arrays', train=False)
#
# trainloader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
#
#
# status_loop = tqdm(trainloader, total=len(trainloader), leave=True)
#
# i = 0
# for x in status_loop:
#     # print(len(x), x[0].shape, x[1].shape)
#     # if i == 3:
#     #     break
#     # i += 1
#     pass
#
#
# ds = CocoDataset(annotations='../coco2017/annotations/instances_train2017.json', image_root_dir='../coco2017', mask_root_dir='../cat_id_masked_arrays', train=True)
#
# trainloader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
#
# status_loop = tqdm(trainloader, total=len(trainloader), leave=True)
#
# i = 0
# for x in status_loop:
#     # print(len(x), x[0].shape, x[1].shape)
#     # if i == 3:
#     #     break
#     # i += 1
#     pass
#
#
#
