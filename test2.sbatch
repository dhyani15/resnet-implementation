#!/bin/bash
# 
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                
#SBATCH --time=5:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=results
#SBATCH --output=test2.out
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=nhd7682@nyu.edu

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate condaenv
python test2.py