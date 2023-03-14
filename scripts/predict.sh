#!/bin/bash
#SBATCH --job-name=prediction
#SBATCH -A kumargroup_gpu
#SBATCH --mem=128G
#SBATCH -p gpu
#SBATCH -C gpu32g
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 24:00:00
#SBATCH -o llps-v2/output/%x-%j.out
#SBATCH -e llps-v2/output/%x-%j.err

source ~/my_env/bin/activate

# model_name dataset output_dir
# $1 model_name
# $2 dataset
# $3 output_dir (default llps-v2/predictions/)
python3 llps-v2/make_predictions.py $1 $2