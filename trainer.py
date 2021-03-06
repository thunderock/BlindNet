# @Filename:    trainer.py.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/23/22 2:45 AM

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from modules.blind_net import BlindNet
from data import data_loader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb
YEAR = 2022

os.environ['PYTHONHASHSEED'] = str(YEAR)
np.random.seed(YEAR)
torch.manual_seed(YEAR)
torch.cuda.manual_seed(YEAR)
torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self,  learning_rate=5e-4, epochs=250, batch_size=16, val_split=.3, image_size=84):

        # Define hparams here or load them from a config file
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("DEVICE " , self.device)
        self.image_size = image_size
        weights = [2.54125732e+10, 2.12089510e+09, 3.50763520e+07, 1.33508720e+08,
 1.15007976e+08, 7.72404640e+07, 1.84975376e+08, 1.78243536e+08,
 1.47735264e+08, 6.05195120e+07, 1.69553120e+07, 2.70199440e+07,
 0.10000000e+00, 3.03079300e+07, 1.72061020e+07, 7.93474800e+07,
 4.12124480e+07, 1.50554320e+08, 1.10039328e+08, 8.80142400e+07,
 4.54385280e+07, 6.46494320e+07, 1.19434568e+08, 4.67601960e+07,
 7.97816160e+07, 8.14715520e+07, 0.10000000e+00, 2.11074160e+07,
 7.49813280e+07, 0.10000000e+00, 0.10000000e+00, 2.37101200e+07,
 6.87035200e+06, 7.44440080e+07, 8.11525100e+06, 6.50637000e+06,
 8.26023700e+06, 3.15884600e+06, 1.68423280e+07, 3.58903500e+06,
 4.36216900e+06, 1.37788370e+07, 2.95647240e+07, 1.39383830e+07,
 3.74215840e+07, 0.10000000e+00, 2.13482820e+07, 5.90344320e+07,
 6.02634800e+06, 9.11835700e+06, 5.36413000e+06, 1.45087296e+08,
 5.49896080e+07, 2.59026360e+07, 7.48723280e+07, 3.26316280e+07,
 4.53098080e+07, 1.75603360e+07, 3.19278560e+07, 1.81366304e+08,
 4.56704320e+07, 8.99498880e+07, 1.39190304e+08, 1.23878712e+08,
 5.68712560e+07, 2.27664032e+08, 0.10000000e+00, 6.82183552e+08,
 0.10000000e+00, 0.10000000e+00, 6.85072480e+07, 0.10000000e+00,
 8.44416480e+07, 8.72198400e+07, 4.25970900e+06, 1.30164830e+07,
 3.32516220e+07, 2.41753680e+07, 2.24938000e+07, 8.42300560e+07,
 1.55816700e+06, 3.73558600e+07, 8.72179680e+07, 0.10000000e+00,
 4.07338000e+07, 3.41035440e+07, 3.79678160e+07, 8.55178600e+06,
 7.85969840e+07, 8.44164000e+05, 3.70332300e+06]
        weights = [1 - (x / sum(weights)) for x in weights]
        self.weights = torch.FloatTensor(weights).to(self.device)

    def train(self, train_loader, model, criterion, optimizer, epoch):
        # Training loop begin
        running_loss = 0
        model.train()
        status_loop = tqdm(train_loader, total=len(train_loader), leave=False)
        for i, data in enumerate(status_loop):
            # Get the inputs
            inputs, labels, random_cat = data
            # print(torch.unique(labels))
            # Move them to the correct device
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)

            labels = labels.reshape(-1)
            # print(outputs[1050, :])
            # outputs = torch.max(outputs, dim=1)[1]
            # assert torch.sum(labels, axis=1).sum() == 1
            # print(outputs.shape, outputs.shape, random_cat)
            # Compute the loss
            loss = criterion(outputs, labels.to(torch.long))
            # Backward pass
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Add the loss to the running loss
            running_loss += loss.item()
            # Every 20 iterations, print the loss
            # if i % 50 == 49:
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 50))
            #     running_loss = 0.0
            status_loop.set_description('Training, epoch: %d' % (epoch + 1))
            status_loop.set_postfix(loss=loss.item())

        return running_loss

    def validate(self, val_loader, model, criterion):
        model.eval()

        running_loss = 0

        for i, data in enumerate(tqdm(val_loader, leave=False)):
            images, labels, _ = data
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            running_loss += criterion(outputs, labels.reshape(-1).to(torch.long))
        avg_loss = running_loss / len(val_loader)
        print('Validation Loss: %.3f' % avg_loss)
        return avg_loss

    def train_and_evaluate(self, model_save_name, scheduler, optimizer, model, data_dir=""):
        # dataloaders
        dataset = data_loader.CocoDataset(annotations='{}/coco2017/annotations/instances_train2017.json'.format(data_dir),
                                          image_root_dir='{}/coco2017'.format(data_dir), mask_root_dir='{}/cat_id_masked_arrays'.format(data_dir), train=True, image_size=self.image_size)
        img_idxs = dataset.img_ids
        val_split = int(len(img_idxs) * self.val_split)
        # print(val_split, self.val_split)
        trainset, validset = random_split(dataset, [len(img_idxs) - val_split, val_split])
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=3)
        if val_split > 0:
            validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=True, num_workers=3)


        # for i, param in enumerate(model.parameters()):
        #     if i < 10:
        #         param.requires_grad = False
        criterion = nn.CrossEntropyLoss(weight=self.weights)
        # optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        # scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6, verbose=True)
        best_val_loss = np.inf
        # Train and evaluate
        for epoch in range(self.epochs):
            loss = self.train(trainloader, model, criterion, optimizer, epoch)
            if val_split > 0:
                with torch.no_grad():
                    loss = self.validate(validloader, model, criterion)
                    if loss < best_val_loss:
                        print('Saving model..... Loss improved from %.3f to %.3f' % (best_val_loss, loss))
                        best_val_loss = loss
                        torch.save(model.state_dict(), '{}.pt'.format(model_save_name))
                    wandb.log({"validation_loss": loss, "epoch": epoch, "model_name": model_save_name})
            optimizer.step()
            scheduler.step(loss)
            # scheduler2.step(epoch + 1/self.epochs)
        return best_val_loss


class MultiLabelTrainer(Trainer):
    def __init__(self, learning_rate=5e-4, epochs=250, batch_size=16, val_split=.3, image_size=84):
        super().__init__(learning_rate, epochs, batch_size, val_split, image_size)

    def train(self, train_loader, model, criterion, optimizer, epoch):
        # Training loop begin
        running_loss = 0
        model.train()
        status_loop = tqdm(train_loader, total=len(train_loader), leave=False)
        for i, data in enumerate(status_loop):
            # Get the inputs
            inputs, labels, _ = data
            # Move them to the correct device
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Add the loss to the running loss
            running_loss += loss.item()

            status_loop.set_description('Training, epoch: %d' % (epoch + 1))
            status_loop.set_postfix(loss=loss.item())

        return running_loss

    def validate(self, val_loader, model, criterion):
        model.eval()

        running_loss = 0

        for i, data in enumerate(tqdm(val_loader, leave=False)):
            images, labels, _ = data
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            running_loss += criterion(outputs, labels)
        avg_loss = running_loss / len(val_loader)
        print('Validation Loss: %.3f' % avg_loss)
        return avg_loss

    def train_and_evaluate(self, model_save_name, scheduler, optimizer, model, data_dir=""):
        # dataloaders
        dataset = data_loader.CocoDataset(annotations='{}/coco2017/annotations/instances_train2017.json'.format(data_dir),
                                          image_root_dir='{}/coco2017'.format(data_dir), mask_root_dir='{}/cat_id_masked_arrays'.format(data_dir), train=True, image_size=self.image_size,
                                          multilabel=True)
        img_idxs = dataset.img_ids
        val_split = int(len(img_idxs) * self.val_split)
        # print(val_split, self.val_split)
        trainset, validset = random_split(dataset, [len(img_idxs) - val_split, val_split])
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=6)
        if val_split > 0:
            validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=True, num_workers=6)


        # for i, param in enumerate(model.parameters()):
        #     if i < 10:
        #         param.requires_grad = False
        criterion = nn.BCEWithLogitsLoss(weight=self.weights)
        # optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        # scheduler2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6, verbose=True)
        best_val_loss = np.inf
        # Train and evaluate
        for epoch in range(self.epochs):
            loss = self.train(trainloader, model, criterion, optimizer, epoch)
            if val_split > 0:
                with torch.no_grad():
                    loss = self.validate(validloader, model, criterion)
                    if loss < best_val_loss:
                        print('Saving model..... Loss improved from %.3f to %.3f' % (best_val_loss, loss))
                        best_val_loss = loss
                        torch.save(model.state_dict(), '{}.pt'.format(model_save_name))
                    wandb.log({"validation_loss": loss, "epoch": epoch, "model_name": model_save_name})
            optimizer.step()
            scheduler.step(loss)
            # scheduler2.step(epoch + 1/self.epochs)
        return best_val_loss

# model = Trainer(image_size=32)
# model.train_and_evaluate("model_name)
