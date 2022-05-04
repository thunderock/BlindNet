# @Filename:    driver.py.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        5/3/22 12:49 AM
from sweep_tuner import SweepTuner
import wandb
wandb.init(project="blindnet_new_weights_carbonate")
trainer = SweepTuner(image_size=64)
trainer.train_and_evaluate(
    model="resnet18",
    learning_rate=0.0006197,
    batch_size=64,
    optimizer="adam",
    scheduler="ReduceLROnPlateau",
    name="temp",
    data_dir='.'
)