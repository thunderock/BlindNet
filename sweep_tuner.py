# @Filename:    sweep_tuner.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/24/22 8:02 PM
import gc

import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from data.data_loader import CocoDataset
from torch.utils.data import DataLoader, random_split
from modules.blind_net_fft import BlindNetFFT
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from trainer import Trainer
from tqdm import tqdm
from torch import optim
from PIL import Image, ImageOps
import wandb


sweep_config = {
    'method': 'bayes'
}


parameters_dict = {
    'optimizer': {
        'values': ['adam' ,'sgd']
        },
    'scheduler':{
        'values': ['CosineAnnealingLR', 'ReduceLROnPlateau']

    },
    'model': {"values": ['resnet18']}  # , 'resnet50']}
}

sweep_config['parameters'] = parameters_dict

metric = {
    'name': 'validation_loss',
    'goal': 'minimize'
    }
sweep_config['metric'] = metric
parameters_dict.update({
    'learning_rate': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.01
      },
    'batch_size':
        {"values": [8, 16, 32]}

    })

YEAR = 2022
os.environ['PYTHONHASHSEED'] = str(YEAR)
np.random.seed(YEAR)
torch.manual_seed(YEAR)
torch.cuda.manual_seed(YEAR)
torch.backends.cudnn.deterministic = True

class Config:
    seed = 42
    verbose = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True
    k = 5
    selected_folds = [0]
    input_dim = 5
    dense_dim = 512
    logit_dim = 512
    num_classes = 1
    epochs =150
    warmup_prop = 0
    T_max=50
    T_0=50
    min_lr=1e-6
    num_cycles=0.5
    val_bs = 256
    first_epoch_eval = 0


class SweepTuner:
    def __init__(self, image_size=84):

        # Define hparams here or load them from a config file
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size


    def get_optimizer(self, model, method, learning_rate):
        if method == 'adam':
            return optim.Adam(model.parameters(), lr=learning_rate)
        elif method == 'sgd':
            return optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError('Unknown optimizer method')


    def get_scheduler(self, optimizer, method):
        if method == 'CosineAnnealingLR':
            return CosineAnnealingLR(optimizer, T_max=Config.T_max, eta_min=Config.min_lr, last_epoch=-1)
        if method == 'CosineAnnealingWarmRestarts':
            return CosineAnnealingWarmRestarts(optimizer, T_0=Config.T_0, T_mult=1, eta_min=Config.min_lr, last_epoch=-1)
        if method == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(optimizer, patience=3, verbose=False)
        else:
            raise ValueError('Unknown scheduler method')

    def train_and_evaluate(self, model, learning_rate, batch_size, optimizer, scheduler, data_dir="."):
        model = BlindNetFFT(image_size=self.image_size, model_name=model).to(self.device)
        optimizer = self.get_optimizer(model, method=optimizer, learning_rate=learning_rate)
        scheduler = self.get_scheduler(optimizer, method=scheduler)
        trainer = Trainer(learning_rate=learning_rate, batch_size=batch_size, image_size=self.image_size,)
        trainer.train_and_evaluate(model=model, model_save_name="{}_{}_{}".format(model, learning_rate, batch_size),scheduler=scheduler, optimizer=optimizer,data_dir=data_dir)
        del (trainer)
        gc.collect()
        torch.cuda.empty_cache()


def train(con=None):
    with wandb.init(config=con):
        con = wandb.config
        print(con)

        trainer = SweepTuner()

        trainer.train_and_evaluate(
            model=con.model,
            learning_rate=con.learning_rate,
            batch_size=con.batch_size,
            optimizer=con.optimizer,
            scheduler=con.scheduler
            )


    gc.collect()
    torch.cuda.empty_cache()


# sweep_id = wandb.sweep(sweep_config, project="sweeps_blindnet")
# run = wandb.agent(sweep_id, train, count=20)
# trainner = SweepTuner(image_size=32)
# trainner.train_and_evaluate(
#     model="resnet18",
#     learning_rate=1e-3,
#     batch_size=16,
#     optimizer="adam",
#     scheduler="CosineAnnealingWarmRestarts"
# )