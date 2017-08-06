#!/bin/bash

#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -n 1
#SBATCH --mem=55gb
#SBATCH --gres=gpu:2
#SBATCH -t 4-23:00:00
#SBATCH -J densenet_1000
#SBATCH -o my_output%j
#SBATCH --mail-type=begin
#SBATCH --mail-user=npatil4@ur.rochester.edu

module load intelpython
module load tensorflow
python ./densenet.py