import json

import numpy as np
import random

import torch
from PIL import Image
from matplotlib import pyplot as plt
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def read_annots_bbox(annot_path):
    with open(annot_path, "r") as f:
        data = json.load(f)
        for k in data["annotations"]:
            del k["segmentation"]

    return data

def create_img_meta(data):
    img_annots = {k["id"]:k for k in data["images"]}
    for k in img_annots:
        img_annots[k]["bboxes"] = []
        img_annots[k]["cats"] = []

    img_ids = [k["id"] for k in data["images"]]

    cats = data["categories"]
    for k in data["annotations"]:
        img_annots[k["image_id"]]["bboxes"].append(k["bbox"])
        img_annots[k["image_id"]]["cats"].append(k["category_id"])

    for k in img_annots:
        img_annots[k]["bboxes"] = torch.tensor((img_annots[k]["bboxes"])).long()

    return img_annots, img_ids, cats

def create_img_meta_from_path(annot_path):
    data = read_annots_bbox(annot_path)
    return create_img_meta(data)

def mask_img_with_bbox(img, bbox, fill_value = 0):
    bbox = bbox.long()
    img[:, bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]] = fill_value
    return img

def mask_img(src_img, src_bboxes, tar_img, tar_bboxes, cats, total_categories = 80):
    tar_masked_cls = torch.zeros(total_categories, tar_img.shape[1], tar_img.shape[1])
    masked_bbox = torch.zeros(1, src_img.shape[1], src_img.shape[1])
    if src_bboxes.shape[0]==0:
        return src_img, tar_masked_cls, masked_bbox, torch.zeros(4).long(), -1

    masked_idx = random.randrange(src_bboxes.shape[0])

    src_masked_box, tar_masked_box = src_bboxes[masked_idx], tar_bboxes[masked_idx]
    src_img = mask_img_with_bbox(src_img, src_masked_box, fill_value=0)
    masked_bbox = mask_img_with_bbox(masked_bbox, tar_masked_box, fill_value=1)
    for cls, bbox in zip(cats, tar_bboxes):
        tar_masked_cls[cls, bbox[0]:bbox[0]+bbox[2], bbox[1]:bbox[1]+bbox[3]] = 1

    return src_img, tar_masked_cls, masked_bbox, tar_masked_box, cats[masked_idx]

def read_img(idx, img_ids, img_annots, img_dir_path):
    annot_id = img_ids[idx]
    filename = img_annots[annot_id]["file_name"]
    path = img_dir_path + filename
    return np.array(Image.open(path).convert("RGB")), img_annots[annot_id]

def mask_object(img, bboxes, categories, target_dims):
    pass

#Code inspired from: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/a2ee9271b5280be6994660c7982d0f44c67c3b63/ML/Pytorch/Basics/albumentations_tutorial/utils.py
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

def get_val_transformation(config):
    transes = []
    IMG_SIZE = config["inp_img_size"]  # W = H
    transes.append(A.LongestMaxSize(max_size=int(IMG_SIZE)))
    transes.append(A.PadIfNeeded(min_height=int(IMG_SIZE), min_width=int(IMG_SIZE), border_mode=0, value=(0,0,0)))
    transes.append(A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ))
    transes.append(ToTensorV2())

    # Apply this transformation to get the bboxes for the target image
    inp_tar_transformations = []
    TARGET_SIZE = config["target_img_size"]
    inp_tar_transformations.append(A.Resize(height=TARGET_SIZE, width=TARGET_SIZE))
    inp_tar_transformations.append(ToTensorV2())

    return A.Compose(transes, bbox_params=A.BboxParams(format="coco", label_fields=[]), ), A.Compose(
        inp_tar_transformations, bbox_params=A.BboxParams(format="coco", label_fields=[]))

def get_transformation(config):
    transes = []
    IMG_SIZE = config["inp_img_size"] # W = H
    scale = config["inp_img_scale"]
    transes.append(A.LongestMaxSize(max_size=int(IMG_SIZE*scale)))
    transes.append(A.PadIfNeeded(min_height=int(IMG_SIZE * scale), min_width=int(IMG_SIZE * scale), border_mode=0, value=(0,0,0)))
    transes.append(A.RandomCrop(width=IMG_SIZE, height=IMG_SIZE))
    transes.append(A.OneOf([
            A.ShiftScaleRotate(rotate_limit=10, p=0.2, border_mode=cv2.BORDER_CONSTANT),
            A.Affine(shear=15, p=0.1),
        ], p=1.0, ))
    transes.append(A.HorizontalFlip(p=0.5))
    transes.append(A.OneOf([A.Blur(blur_limit=2, p=0.1), A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4)], p=0.2))
    transes.append(A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5))
    transes.append(A.ChannelShuffle(p=0.05))
    transes.append(A.CLAHE(p=0.1))
    transes.append(A.Posterize(p=0.1))
    transes.append(A.ToGray(p=0.1))
    transes.append(A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,))
    transes.append(ToTensorV2())

    # Apply this transformation to get the bboxes for the target image
    inp_tar_transformations = []
    TARGET_SIZE = config["target_img_size"]
    inp_tar_transformations.append(A.Resize(height=TARGET_SIZE, width=TARGET_SIZE))
    inp_tar_transformations.append(A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ))
    inp_tar_transformations.append(ToTensorV2())

    return A.Compose(transes, bbox_params=A.BboxParams(format="coco", label_fields=[]),), A.Compose(inp_tar_transformations, bbox_params=A.BboxParams(format="coco", label_fields=[]))

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

def apply_sample_transformations(annot_path, img_dir):
    config = {
        "inp_img_size": 500,
        "inp_img_scale": 1.1,
        "target_img_size": 300
    }

    img_annot, img_ids, cats = create_img_meta_from_path(annot_path)
    for k in range(100):
        img_idx = random.choice(range(len(img_ids)))
        # img_idx = 3421
        print(f"Chosen img_id: {img_ids[img_idx]}, {img_idx}")
        img, img_meta = read_img(img_idx, img_ids, img_annot, img_dir)
        bboxes = img_meta["bboxes"]

        input_transform, target_transform = get_transformation(config)

        all_images = [img]
        all_bboxes = [bboxes[0]]

        for i in range(5):
            augmentations = input_transform(image=img, bboxes=bboxes.long())
            aug_img = augmentations["image"]
            aug_img = aug_img.permute(1,2,0)

            aug_bboxes = np.array(augmentations["bboxes"]).astype(np.int16)
            all_images.append((aug_img.numpy()*255).astype(np.uint8))
            all_bboxes.append(aug_bboxes[0] if len(aug_bboxes)>=1 else None)

            # Add transformed image
            aug_bboxes[:, -2:] = np.clip(aug_bboxes[:, -2:], 1, None)
            aug_img_tar = (aug_img*255).numpy().astype(np.uint8)

            try:
                tar_augmentations = target_transform(image = aug_img_tar, bboxes = aug_bboxes)
            except Exception as e:
                print(aug_bboxes)
                raise e
            aug_img_tar = tar_augmentations["image"]
            aug_bboxes_tar = tar_augmentations["bboxes"]
            all_images.append((aug_img_tar.permute(1,2,0).numpy()*255).astype(np.uint8))
            all_bboxes.append(aug_bboxes_tar[0] if len(aug_bboxes) >= 1 else None)

            if len(aug_bboxes) == 0:
                print("Augmented bbox size is zero")

        plot_examples(all_images, all_bboxes)
    # plot_examples(all_images, None)


if __name__ == '__main__':
    apply_sample_transformations("datasets/coco/annotations/instances_val2017.json", "datasets/coco/val2017/")
