import random

import torch
from torch.utils.data import Dataset, DataLoader
import  numpy as np
from tqdm import tqdm

from src.img_utils import create_img_meta_from_path, get_transformation, get_val_transformation, read_img, mask_img, \
    plot_examples


class SingleMaskCocoDataset(Dataset):
    def __init__(self, annot_path, img_dir_path, config, train_dataset = True):
        self.img_annots, self.img_ids, self.categories = create_img_meta_from_path(annot_path)
        self.img_dir_path = img_dir_path
        self.config = config
        self.src_transformation, self.target_transformation = get_transformation(config) if train_dataset else get_val_transformation(config)
        self.total_categories = config["total_categories"]
        self.score_mode = False

    def __len__(self):
        return len(self.img_ids)

    def get_image_by_id(self, img_id, masked = False):
        for i in range(len(self.img_ids)):
            if str(self.img_ids[i]) == str(img_id):
                annot_id = self.img_ids[i]
                saved_boxes = self.img_annots[annot_id]["bboxes"]
                if not masked:
                    self.img_annots[annot_id]["bboxes"] = torch.tensor([])
                img, cls =  self.__getitem__(i)
                if not masked:
                    self.img_annots[annot_id]["bboxes"] = saved_boxes
                return img, cls, self.categories

        raise NotImplementedError("ITEM ID NOT FOUND")

    def __getitem__(self, item):
        img, img_meta = read_img(item, self.img_ids, self.img_annots, self.img_dir_path)
        bboxes = img_meta["bboxes"]
        if bboxes.shape[0] > 0:
            bboxes[:, -2:] = torch.clip(bboxes[:, -2:], 1, None)
        src_augs = self.src_transformation(image = img, bboxes = bboxes)

        src_img = src_augs["image"]
        src_bboxes = torch.tensor(src_augs["bboxes"]).long()

        if src_bboxes.shape[0] > 0:
            src_bboxes[:, -2:] = torch.clip(src_bboxes[:, -2:], 1, None)


        if src_bboxes.shape[0] == 0:
            return torch.cat((src_img, torch.zeros(1, *src_img.shape[-2:])), dim=0), torch.tensor(-1)

        masked_idx = random.randrange(src_bboxes.shape[0])
        src_bbox = src_bboxes[masked_idx]
        masked_channel = torch.zeros(1, *src_img.shape[-2:])
        masked_channel[:, src_bbox[0]:src_bbox[0]+src_bbox[2], src_bbox[1]: src_bbox[1]+src_bbox[3]] = 1
        # src_img[:, src_bbox[0]:src_bbox[0]+src_bbox[2], src_bbox[1]: src_bbox[1]+src_bbox[3]] = 0
        src_img[:, src_bbox[1]: src_bbox[1] + src_bbox[3], src_bbox[0]:src_bbox[0] + src_bbox[2]] = 0
        return torch.cat((src_img, masked_channel), dim=0), torch.tensor(img_meta["cats"][masked_idx])

        # src_img, tar_masked_cls, masked_bbox_channel, tar_masked_bbox, cat = mask_img(src_img, src_bboxes, tar_img, tar_bboxes, img_meta["cats"], self.total_categories)
        # inp_img = torch.cat((src_img, masked_bbox_channel), dim=0)
        #
        # if not self.score_mode:
        #     return inp_img, tar_masked_cls
        # else:
        #     return inp_img, tar_masked_cls, tar_masked_bbox, torch.tensor([cat])

        # TODO: Augmentations

        # Read the image
        # Get the bbox
        # Mask an object
        # Create the input
        # Create the target

if __name__ == '__main__':
    config = {
        "annotation_path": "datasets/coco/annotations/instances_val2017.json",
        "img_dir_path": "datasets/coco/val2017/",
        "inp_img_size": 384,
        "inp_img_scale": 1.1,
        "target_img_size": 384,
        "total_categories": 91
    }
    dataset = SingleMaskCocoDataset(config["annotation_path"], config["img_dir_path"], config, train_dataset=False)
    dataset.score_mode = True
    loader = DataLoader(dataset = dataset, batch_size = 2, shuffle = False, num_workers=4)

    print(len(dataset))
    num_iter = 0
    for i, (inp_img, tar_cls) in tqdm(enumerate(loader)):
        num_iter += 1
        print(num_iter)
        assert inp_img.shape[-1] == config["inp_img_size"]
        assert inp_img.shape[-2] == config["inp_img_size"]

        # if (i+1)%100 == 0:
        #     plot_examples([inp_img[0].permute(1,2,0)[...,:3].numpy()])

    print("Done")
