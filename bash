#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=blazor
#SBATCH --partition=dl
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node p100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=64gb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashutiwa@iu.edu


conda deactivate
module load deeplearning/2.8.0
srun python driver.py & python driver_multilabel.py
