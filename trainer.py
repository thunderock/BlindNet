# @Filename:    trainer.py.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/23/22 2:45 AM

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from modules import BlindNetFFT
from data import data_loader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
YEAR = 2022

os.environ['PYTHONHASHSEED'] = str(YEAR)
np.random.seed(YEAR)
torch.manual_seed(YEAR)
torch.cuda.manual_seed(YEAR)
torch.backends.cudnn.deterministic = True

class Trainer:
    def __init__(self,  learning_rate=5e-4, epochs=35, batch_size=16, val_split=.2):

        # Define hparams here or load them from a config file
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, train_loader, model, criterion, optimizer, epoch):
        # Training loop begin
        running_loss = 0
        model.train()
        status_loop = tqdm(train_loader, total=len(train_loader), leave=True)
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
            loss = criterion(F.one_hot(outputs, 91), F.one_hot(labels, 91))
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

        for i, data in enumerate(tqdm(val_loader)):
            images, labels, _ = data
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            running_loss += criterion(outputs, labels)
        avg_loss = running_loss / len(val_loader)
        print('Validation Loss: %.3f' % avg_loss)
        return avg_loss

    def train_and_evaluate(self):
        # dataloaders
        dataset = data_loader.CocoDataset(annotations='coco2017/annotations/instances_val2017.json',
                                          image_root_dir='coco2017', mask_root_dir='cat_id_masked_arrays', train=False)
        img_idxs = dataset.img_ids
        val_split = int(len(img_idxs) * self.val_split)
        # print(val_split, self.val_split)
        trainset, validset = random_split(dataset, [len(img_idxs) - val_split, val_split])
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        if val_split > 0:
            validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        model = BlindNetFFT().to(self.device)
        # for i, param in enumerate(model.parameters()):
        #     if i < 10:
        #         param.requires_grad = False
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
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
            optimizer.step()
            scheduler1.step(loss)
            # scheduler2.step(epoch + 1/self.epochs)
        torch.save(model.state_dict(), 'model.pt')

