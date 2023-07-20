#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --job-name=train-model
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

# train.py num_epochs learning_rate batch_size dropout training_file model_nametag
python3 train.py $1 1e-4 $2 0.3 $3 $4
