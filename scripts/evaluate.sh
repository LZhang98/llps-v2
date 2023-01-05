#!/bin/bash
#SBATCH --job-name=evaluate-model
#SBATCH -A kumargroup_gpu
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 24:00:00
#SBATCH -o llps-v2/output/%x-%j.out
#SBATCH -e llps-v2/output/%x-%j.err

source ~/my_env/bin/activate

# model_name dataset batch_size threshold logfile
python3 llps-v2/evaluate.py 2023-01-04_full_e200_lr-4_dropout-0.3 llps-v2/data/test_set_1_pos.csv 8 2000 2023-01-04-eval