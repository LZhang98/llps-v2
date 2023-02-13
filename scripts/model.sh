#!/bin/bash
#SBATCH --job-name=model
#SBATCH -A kumargroup_gpu
#SBATCH --mem=16G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 1:00:00
#SBATCH -o llps-v2/output/%x-%j.out
#SBATCH -e llps-v2/output/%x-%j.err

source ~/my_env/bin/activate

python3 llps-v2/model.py