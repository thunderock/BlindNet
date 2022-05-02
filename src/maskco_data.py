import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from src.img_utils import get_transformation, create_img_meta_from_path, read_img, mask_img, get_val_transformation

class MaskCocoDataset(Dataset):
    def __init__(self, annot_path, img_dir_path, config, train_dataset = True):
        self.img_annots, self.img_ids, self.categories = create_img_meta_from_path(annot_path)
        self.img_dir_path = img_dir_path
        self.config = config
        self.src_transformation, self.target_transformation = get_transformation(config) if train_dataset else get_val_transformation(config)
        self.total_categories = config["total_categories"]
        self.score_mode = False

    def __len__(self):
        return len(self.img_ids)

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

        # See if multiplying by 255 is necessary
        tar_augs = self.target_transformation(image = (src_img*255).long().permute(1,2,0).numpy().astype(np.uint8), bboxes = src_bboxes)
        tar_img = tar_augs["image"]
        tar_bboxes = torch.tensor(tar_augs["bboxes"]).long()

        src_img, tar_masked_cls, masked_bbox_channel, tar_masked_bbox, cat = mask_img(src_img, src_bboxes, tar_img, tar_bboxes, img_meta["cats"], self.total_categories)
        inp_img = torch.cat((src_img, masked_bbox_channel), dim=0)

        if not self.score_mode:
            return inp_img, tar_masked_cls
        else:
            return inp_img, tar_masked_cls, tar_masked_bbox, torch.tensor([cat])

        # TODO: Augmentations

        # Read the image
        # Get the bbox
        # Mask an object
        # Create the input
        # Create the target

def visualize_bbox(img, bboxes, class_name, color=(255, 0, 0), thickness=5):
    """Visualizes a single bounding box on the image"""
    # for bbox in bboxes:
    x_min, y_min, width, height = map(int, bboxes)
    try:
        cv2.rectangle(img, (x_min, y_min), (x_min+width, y_min+height), color, thickness)
    except Exception as e:
        print(e.msg)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.rectangle(img, (x_min, y_min), (x_min + width, y_min + height), color, thickness)
    return img

from matplotlib import pyplot as plt

def plot_examples(images, bboxes=None):
    fig = plt.figure(figsize=(15, 15))
    columns = 4
    rows = 5

    for i in range(1, len(images)+1):
        if bboxes is not None:
            img = visualize_bbox(images[i - 1], bboxes[i - 1], class_name="Elon")
        else:
            img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    config = {
        "annotation_path": "datasets/coco/annotations/instances_val2017.json",
        "img_dir_path": "datasets/coco/val2017/",
        "inp_img_size": 384,
        "inp_img_scale": 1.1,
        "target_img_size": 384,
        "total_categories": 91
    }
    dataset = MaskCocoDataset(config["annotation_path"], config["img_dir_path"], config, train_dataset=False)
    dataset.score_mode = True
    loader = DataLoader(dataset = dataset, batch_size = 2, shuffle = False)

    print(len(dataset))
    num_iter = 0
    for inp_img, tar_cls, bbox, cats in tqdm(loader):
        num_iter += 1
        print(num_iter)
        assert inp_img.shape[-1] == config["inp_img_size"]
        assert inp_img.shape[-2] == config["inp_img_size"]
        assert tar_cls.shape[-1] == config["inp_img_size"]
        assert tar_cls.shape[-2] == config["inp_img_size"]

        # if (i+1)%100 == 0:
        #     plot_examples([inp_img.permute(1,2,0)[...,:3].numpy()])

    print("Done")
