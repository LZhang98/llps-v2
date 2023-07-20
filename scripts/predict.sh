#!/bin/bash
#SBATCH --job-name=prediction
#SBATCH --account=def-sushant
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH -o /home/lzhang98/scratch/slurm-output/%x-%j.out
#SBATCH -e /home/lzhang98/scratch/slurm-output/%x-%j.err
#SBATCH --mail-user=luke.zhang@uhn.ca
#SBATCH --mail-type=ALL

cd /home/lzhang98/
source my_env3.9/bin/activate
cd projects/def-sushant/lzhang98/llps-v2

# model_name dataset output_dir
# $1 model_name
# $2 dataset
# $3 output_name
# $4 model_type ('og' or 'mhsa')
# $5  (default llps-v2/predictions/)
python3 make_predictions.py $1 $2 $3 $4