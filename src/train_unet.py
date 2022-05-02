import os
import traceback

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.maskco_data import MaskCocoDataset
from src.model.unet_model import UNet
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_bbox_dataloader(annoation_path, img_dir_path, config, train = True):
    dataset = MaskCocoDataset(annoation_path, img_dir_path, config, train_dataset= train)
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

        model.load_state_dict(torch.load(model_path))
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
    scores = { "total": 0,"pred_idx_mask": [0 for i in range(91)], "pred_idx": [0 for i in range(91)] }

    logging.info("Calculating scores.")
    model.eval()
    loader.dataset.score_mode = True
    total_loss = 0

    with torch.no_grad():
        for inp_img, masked_labels, bboxes, cats in loader:
            inp_img = inp_img.to(device)
            masked_labels = masked_labels.to(device)
            masked_labels_pred = model(inp_img)
            loss = loss_fn(masked_labels_pred, masked_labels)
            total_loss += loss.item()

            for i in range(inp_img.shape[0]):
                if cats[i].item() == -1:
                    continue
                scores["total"] += 1
                box_cls_pred = masked_labels_pred[i][:, bboxes[i][0]: bboxes[i][0]+bboxes[i][2], bboxes[i][1]: bboxes[i][0]+bboxes[i][3]].sum(1).sum(1)
                scores["pred_idx_mask"][torch.argsort(box_cls_pred)[cats[i].item()].item()] += 1

    loader.dataset.score_mode = False
    return scores, total_loss

def train_model(config):
    model = UNet(config["init_channels"], config["total_categories"], config["bilinear"])
    model.to(device)

    if config["load_model"]:
        load_model(model, config["model_path"])

    train_loader, val_loader = get_bbox_train_val_loaders(config)
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(), lr=config["lr"])

    epoch_loss = {"train": [], "val": []}

    for epochs in range(config["epochs"]):
        train_loss = 0
        model.train()

        for batch_idx, (input_img, masked_labels) in tqdm(enumerate(train_loader)):
            input_img = input_img.to(device)
            masked_labels = masked_labels.to(device)

            masked_labels_pred = model(input_img)

            loss = loss_fn(masked_labels_pred, masked_labels)
            loss = loss / config["grad_acc"]
            loss.backward()

            if batch_idx == (len(train_loader) - 1) or (batch_idx+1)%config["grad_acc"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

        epoch_loss["train"].append(train_loss / len(train_loader))
        scores, val_loss = calc_scores(model, val_loader, loss_fn)
        epoch_loss["val"].append(val_loss / len(val_loader))

        logging.info(f"Scores: {scores}")
        logging.info(f"train_loss: {train_loss}, val_loss: {val_loss}")

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
        "load_model": False,
        "model_path": "saved_models/",
        "model_type": "unet",
        "bilinear": False,
        "init_channels": 4,
        "bbox": True,
        "batch_size": 20,
        "epochs": 100,
        "lr": 1e-3,
        "log_dir": "logs/",
        "grad_acc": 4
    }

    model_path = f"unet_{config['model_type']}_{'bilinear' if config['bilinear'] else 'transpose'}_{'bbox' if config['bbox'] else '$$'}_{config['inp_img_size']}_{config['batch_size']}.model"
    config["model_path"] += model_path

    log_file_name = f"{config['log_dir']}{model_path}.txt"
    print(log_file_name)
    os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
    logging.basicConfig(filename=log_file_name, filemode="a", format='%(asctime)s %(levelname)s %(message)s',datefmt='%H:%M:%S', level=logging.DEBUG)
    logging.info("Training begun")
    train_model(config)