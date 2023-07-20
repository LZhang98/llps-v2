#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --job-name=evaluation
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -t 1:00:00
#SBATCH -o /home/lzhang98/scratch/slurm-output/%x-%j.out
#SBATCH -e /home/lzhang98/scratch/slurm-output/%x-%j.err
#SBATCH --mail-user=luke.zhang@uhn.ca
#SBATCH --mail-type=ALL

cd /home/lzhang98/
source my_env3.9/bin/activate
cd projects/def-sushant/lzhang98/llps-v2

# model_name dataset batch_size threshold logfile
# $1 model name
# $2 logfile name
# $3 model type ('og' or 'mhsa')
python3 evaluate.py $1 length_matched_test_set.csv 1 -1 $2 $3