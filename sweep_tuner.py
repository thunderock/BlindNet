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
from modules.blind_net import BlindNet
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from trainer import Trainer, MultiLabelTrainer
from tqdm import tqdm
from torch import optim
import warnings

warnings.filterwarnings("ignore")

# from botorch.settings import suppress_botorch_warnings, validate_input_scaling
# import optuna
# from optuna.integration.wandb import WeightsAndBiasesCallback
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
        {"values": [32]}

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
            return ReduceLROnPlateau(optimizer, patience=5, verbose=False)
        else:
            raise ValueError('Unknown scheduler method')

    def train_and_evaluate(self, model, learning_rate, batch_size, optimizer, scheduler, name, data_dir=".", multilabel=False):
        model = BlindNet(image_size=self.image_size, model_name=model, multi_label=multilabel).to(self.device)
        optimizer = self.get_optimizer(model, method=optimizer, learning_rate=learning_rate)
        scheduler = self.get_scheduler(optimizer, method=scheduler)
        trainer = MultiLabelTrainer(learning_rate=learning_rate, batch_size=batch_size, image_size=self.image_size) if multilabel else Trainer(learning_rate=learning_rate, batch_size=batch_size, image_size=self.image_size)
        loss = trainer.train_and_evaluate(model=model, model_save_name=name,scheduler=scheduler, optimizer=optimizer,data_dir=data_dir)
        del (trainer)
        gc.collect()
        torch.cuda.empty_cache()
        return loss


def train_sweep(con=None):
    import wandb
    with wandb.init(config=con):
        import string, random
        letters = string.ascii_lowercase
        name = ''.join(random.choice(letters) for i in range(10))
        con = wandb.config
        print(con, name)

        trainer = SweepTuner(image_size=32)

        trainer.train_and_evaluate(
            model=con.model,
            learning_rate=con.learning_rate,
            batch_size=con.batch_size,
            optimizer=con.optimizer,
            scheduler=con.scheduler,
            data_dir='.',
            name=name
            )


    gc.collect()
    torch.cuda.empty_cache()

def _get_trial_values(trial):
    return {
        'model': trial.suggest_categorical('model', ['resnet18']),
        'optimizer': trial.suggest_categorical('optimizer', ['adam']),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'batch_size': 64,
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
        data_dir=data_dir, name=trial.number)


def plot_study(study, path=None):
    plots = [optuna.visualization.matplotlib.plot_parallel_coordinate,
            optuna.visualization.matplotlib.plot_contour,
            optuna.visualization.matplotlib.plot_slice,
            optuna.visualization.matplotlib.plot_param_importances,
            optuna.visualization.matplotlib.plot_edf,
            optuna.visualization.matplotlib.plot_optimization_history]
    for plot in plots:
        try:
            _ = plot(study)
            if path is None:
                plt.show()
            else:
                p = os.path.join(path, plot.__name__ + '.png')
                print("writing fig at ", str(p))

                plt.savefig(p)
        except Exception as e:
            print("Error in plot: ", e)


# wandb_kwargs = {
#                 "project": "blindnet",
#                 "group": "summary",
#                 "job_type": "logging",
#                 "mode": "online"
#                 }
#
# wandbc = WeightsAndBiasesCallback(metric_name="val_loss", wandb_kwargs=wandb_kwargs)
#
# suppress_botorch_warnings(False)
# validate_input_scaling(True)
# study_name = "blindnet_optuna"
# study_dir = "sqlite:///{}/{}.db".format("tune/", study_name)
# sampler = optuna.integration.BoTorchSampler()
# study = optuna.create_study(study_name=study_name, storage=study_dir, direction='minimize', sampler=sampler)


# To run, run these two lines
# study.optimize(lambda trial: train_optuna(trial, data_dir='../input/coco-cat-id-masked-images/'), n_trials=15, n_jobs=2,
#                show_progress_bar=False, callbacks=[wandbc])
# plot_study(study)

# sweep_id = wandb.sweep(sweep_config, project="sweeps_blindnet_carbonate")
# run = wandb.agent(sweep_id, train_sweep, count=20)
