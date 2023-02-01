#!/bin/bash
#SBATCH --job-name=train
#SBATCH -A kumargroup_gpu
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 72:00:00
#SBATCH -o llps-v2/output/%x-%j.out
#SBATCH -e llps-v2/output/%x-%j.err

source ~/my_env/bin/activate

# train.py num_epochs learning_rate batch_size dropout
python3 llps-v2/train.py 300 1e-4 8 0.3
