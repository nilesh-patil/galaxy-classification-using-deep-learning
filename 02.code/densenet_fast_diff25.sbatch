#!/bin/bash

#SBATCH -p gpu
#SBATCH -c 10
#SBATCH -n 1
#SBATCH --mem=60gb
#SBATCH --gres=gpu:2
#SBATCH -t 4-23:00:00
#SBATCH -J HUGE_diffed
#SBATCH -o my_output%j
#SBATCH --mail-type=begin
#SBATCH --mail-user=npatil4@ur.rochester.edu

module load intelpython
module load tensorflow
python ./densenet_fast_diff25.py