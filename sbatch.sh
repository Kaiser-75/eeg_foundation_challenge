#!/bin/bash
#SBATCH -N 1                 # number of nodes
#SBATCH -c 28               # number of "tasks" (cores)
#SBATCH -t 0-12:40:59   # time in d-hh:mm:ss
#SBATCH -p general
#SBATCH -q private
#SBATCH -G a100:1
#SBATCH --mem=48G
#SBATCH -o eegSimCLR_slurm.%j.out
#SBATCH -e eegSimCLR_slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shovito@asu.edu

#SBATCH --export=NONE

module load mamba/latest
source activate eeg_meta

echo ">>>>Starting"
python train_clr.py
echo ">>>>End of training"