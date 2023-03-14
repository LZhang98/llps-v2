#!/bin/bash
#SBATCH --job-name=evaluate-model-gpu32g
#SBATCH -A kumargroup_gpu
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH -C gpu32g
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 24:00:00
#SBATCH -o llps-v2/output/%x-%j.out
#SBATCH -e llps-v2/output/%x-%j.err

source ~/my_env/bin/activate

# model_name dataset batch_size threshold logfile
# $1 model name
# $2 logfile name
python3 llps-v2/evaluate.py $1 test_set_1_pos.csv 1 -1 $2