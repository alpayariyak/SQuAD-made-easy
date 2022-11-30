#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH --mem 10G

python main.py 