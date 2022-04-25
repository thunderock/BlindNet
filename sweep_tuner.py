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
        {"values": [16]}

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
        loss = trainer.train_and_evaluate(model=model, model_save_name="{}_{}_{}".format(model, learning_rate, batch_size),scheduler=scheduler, optimizer=optimizer,data_dir=data_dir)
        del (trainer)
        gc.collect()
        torch.cuda.empty_cache()
        return loss


def train_sweep(con=None):
    with wandb.init(config=con):
        con = wandb.config
        print(con)

        trainer = SweepTuner(image_size=32)

        trainer.train_and_evaluate(
            model=con.model,
            learning_rate=con.learning_rate,
            batch_size=con.batch_size,
            optimizer=con.optimizer,
            scheduler=con.scheduler
            )


    gc.collect()
    torch.cuda.empty_cache()

def _get_trial_values(trial):
    return {
        'model': trial.suggest_categorical('model', ['resnet18']),
        'optimizer': trial.suggest_categorical('optimizer', ['adam']),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0, .01),
        'batch_size': trial.suggest_int('batch_size', 16),
        'scheduler': trial.suggest_categorical('scheduler', ['CosineAnnealingLR', 'ReduceLROnPlateau'])
    }


def train_optuna(trial, data_dir="."):
    model_params = _get_trial_values(trial)
    print(model_params)
    trainer = SweepTuner(image_size=32)
    return trainer.train_and_evaluate(
        model=model_params['model'],
        learning_rate=model_params['learning_rate'],
        batch_size=model_params['batch_size'],
        optimizer=model_params['optimizer'],
        scheduler=model_params['scheduler'],
        data_dir=data_dir)

from botorch.settings import suppress_botorch_warnings, validate_input_scaling
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
wandb_kwargs = {"project": "blindnet"}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)

suppress_botorch_warnings(False)
validate_input_scaling(True)

study_dir = "sqlite:///{}/{}.db".format("tune/", "study_name")
sampler = optuna.integration.BoTorchSampler()
study = optuna.create_study(study_name="study_name", storage="tune", direction='minimize', sampler=sampler)
study.optimize(lambda trial: train_optuna(trial, data_dir='../input/coco-cat-id-masked-images/'), n_trials=10, n_jobs=4,
               show_progress_bar=True, callbacks=[wandbc])

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