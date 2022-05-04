import os
import traceback

import numpy as np
import torch
from PIL import Image
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.maskco_data import MaskCocoDataset
from src.model.single_mask_model import SingleMaskCnn
from src.model.unet_model import UNet
import logging

from src.single_maskco_data import SingleMaskCocoDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_bbox_dataloader(annoation_path, img_dir_path, config, train = True):
    dataset = SingleMaskCocoDataset(annoation_path, img_dir_path, config, train_dataset= train)
    dataloader = DataLoader(dataset=dataset, num_workers=4, shuffle=train, batch_size=config["batch_size"])
    # dataloader = DataLoader(dataset=dataset, shuffle=train, batch_size=config["batch_size"])
    return dataloader

def get_bbox_train_val_loaders(config):
    logging.info("Loading train data")
    train_loader = get_bbox_dataloader(config["train_annotation_path"], config["train_img_dir_path"], config, train = True)
    logging.info("Loading val data")
    val_loader = get_bbox_dataloader(config["val_annotation_path"], config["val_img_dir_path"], config, train = False)
    logging.info("Loading complete")
    return train_loader, val_loader

def save_model(model, model_path):
    directory_path = "/".join(model_path.split("/")[:-1])
    if len(directory_path) > 0:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    torch.save(model.state_dict(), model_path)

def load_model(model, model_path):
    try:
        if not os.path.exists(model_path):
            return model

        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    except Exception as e:
        traceback.print_exc(e)
        print("Error occured while loading, ignoring...")

def calc_val_loss(model, loss_fn, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_img, masked_labels in tqdm(loader):
            input_img = input_img.to(device)
            masked_labels = masked_labels.to(device)
            masked_labels_pred = model(input_img)
            loss = loss_fn(masked_labels_pred, masked_labels)
            total_loss += loss.item()
    return total_loss

def calc_scores(model, loader, loss_fn):

    score = {"total": 0, "pred_count": [0 for i in range(91)]}

    logging.info("Calculating scores.")
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pts = 0

    with torch.no_grad():
        for inp_img, cats in loader:
            inp_img = inp_img.to(device)
            cats = cats.to(device)
            inp_img = inp_img[cats!=-1]
            cats = cats[cats!=-1]
            masked_labels_pred = model(inp_img)
            loss = loss_fn(masked_labels_pred, cats)
            total_loss += loss.item()

            total_pts += inp_img.shape[0]
            total_correct += (torch.argmax(masked_labels_pred, dim=-1) == cats).sum()

            pred_idxs = torch.gather(torch.argsort(masked_labels_pred, dim = 1), 1, cats.reshape(-1, 1)).view(-1)

            for i in range(pred_idxs.shape[0]):
                score["pred_count"][pred_idxs[i].item()] += 1

    return total_correct / (total_pts+1e-3), total_loss, score

def plot_img(config, img_id, train = False):
    img_id = "{:012d}.jpg".format(img_id)
    img = np.array(Image.open(config["train_img_dir_path"] if train else config["val_img_dir_path"] + img_id).convert("RGB"))
    plt.imshow(img)
    plt.show()

def perform_inference(config):
    model = SingleMaskCnn(config["init_channels"], config["total_categories"])
    model.to(device)

    if config["load_model"]:
        model = load_model(model, config["model_path"])

    dataset = SingleMaskCocoDataset(config["val_annotation_path"], config["val_img_dir_path"], config, train_dataset= False)
    categories = dataset.categories
    category_names = {cat["id"]: cat["name"] for cat in categories}
    model.train()
    run_simul = True
    masked_obj = True

    while run_simul:
        # img_id = 37988 # Tennis
        # img_id = 129054 # Zebra
        img_id = 579635
        # plot_img(config, img_id)
        img, cls, categories = dataset.get_image_by_id(img_id, masked_obj)
        plt.imshow(img.permute(1,2,0)[..., :3].numpy())
        plt.show()
        # bbox = [80, 100, 200, 220] # Tennis
        bbox = [260, 300, 70, 130] # Zebra

        if not masked_obj:
            img[:3, bbox[0]:bbox[1], bbox[2]: bbox[3]] = 0
            img[3, bbox[0]:bbox[1], bbox[2]: bbox[3]] = 1

        plt.imshow(img.permute(1,2,0)[..., :3].numpy())
        plt.show()

        plt.imshow(img[3].numpy())
        plt.show()

        img = img.to(device)
        cls_pred = model(img.unsqueeze(0))
        cls_pred_sort = torch.argsort(cls_pred, dim=1, descending=True).view(-1)
        top10cats = []
        for i in range(10):
            if cls_pred_sort[i].item() in category_names:
                top10cats.append(category_names[cls_pred_sort[i].item()])
        print(top10cats)

def train_model(config):
    model = SingleMaskCnn(config["init_channels"], config["total_categories"])
    model.to(device)

    if config["load_model"]:
        load_model(model, config["model_path"])

    train_loader, val_loader = get_bbox_train_val_loaders(config)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=config["lr"])

    epoch_loss = {"train": [], "val": []}

    for epochs in range(config["epochs"]):
        train_loss = 0
        model.train()

        for batch_idx, (input_img, cats) in tqdm(enumerate(train_loader)):
            input_img = input_img.to(device)
            cats = cats.to(device)
            input_img = input_img[cats != -1]
            cats = cats[cats != -1]

            masked_labels_pred = model(input_img)

            loss = loss_fn(masked_labels_pred, cats)
            loss = loss / config["grad_acc"]
            loss.backward()

            if batch_idx == (len(train_loader) - 1) or (batch_idx+1)%config["grad_acc"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

        epoch_loss["train"].append(train_loss / len(train_loader))
        val_accuracy, val_loss, score = calc_scores(model, val_loader, loss_fn)
        epoch_loss["val"].append(val_loss / len(val_loader))

        logging.info(f"accuracy: {val_accuracy}")
        logging.info(f"Score: {score}")
        logging.info(f"train_loss: {train_loss/len(train_loader)}, val_loss: {val_loss/len(val_loader)}")

        if epochs == 0 or val_loss == min(epoch_loss["val"]):
            save_model(model, config["model_path"])

    return model

if __name__ == '__main__':
    config = {
        "val_annotation_path": "datasets/coco/annotations/instances_val2017.json",
        "val_img_dir_path": "datasets/coco/val2017/",
        "train_annotation_path": "datasets/coco/annotations/instances_train2017.json",
        "train_img_dir_path": "datasets/coco/train2017/",
        "inp_img_size": 384,
        "inp_img_scale": 1.1,
        "target_img_size": 384,
        "total_categories": 91,
        "train_model": True,
        "save_model": True,
        "load_model": True,
        "model_path": "saved_models/",
        "model_type": "maskcnn",
        "bilinear": False,
        "init_channels": 4,
        "bbox": True,
        "batch_size": 70,
        "epochs": 100,
        "lr": 1e-3,
        "log_dir": "logs/",
        "grad_acc": 4
    }

    model_path = f"cnn_{config['model_type']}_{'bbox' if config['bbox'] else '$$'}_{config['inp_img_size']}_{config['batch_size']}.model"
    config["model_path"] += model_path

    log_file_name = f"{config['log_dir']}{model_path}.txt"
    print(log_file_name)
    os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
    logging.basicConfig(filename=log_file_name, filemode="a", format='%(asctime)s %(levelname)s %(message)s',datefmt='%H:%M:%S', level=logging.DEBUG)
    logging.info("Training begun")
    train_model(config)
    # perform_inference(config)
