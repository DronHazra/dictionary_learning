#!/bin/bash
#SBATCH --job-name=dict
#SBATCH --gpus=1
#SBATCH --output=/data/max_kaufmann/dictionary_learning/experiment_outputs/slurm_logs/%A.out
#SBATCH --time=720

# NOTE: set the environment in your shell before running this script

date;hostname;id;pwd

python train_script.py

date;