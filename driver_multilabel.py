# @Filename:    driver_multilabel.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        5/3/22 12:51 AM
from sweep_tuner import SweepTuner
import wandb
wandb.init(project="blindnet_multilabel_new_weights_carbonate")
trainer = SweepTuner(image_size=128)
trainer.train_and_evaluate(
    model="resnet18",
    learning_rate=0.0006197,
    batch_size=128,
    optimizer="adam",
    scheduler="ReduceLROnPlateau",
    name="temp_multilabel_carbonate",
    data_dir='.',
    multilabel=True
)