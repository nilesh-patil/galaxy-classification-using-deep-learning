#!/bin/bash

#SBATCH -p standard
#SBATCH -c 10
#SBATCH -n 1
#SBATCH --mem=125gb
#SBATCH --gres=gpu:2
#SBATCH -t 4-23:00:00
#SBATCH -J galactic_distribution
#SBATCH -o my_output%j

module load intelpython
module load tensorflow
python ./convnet08_check.py