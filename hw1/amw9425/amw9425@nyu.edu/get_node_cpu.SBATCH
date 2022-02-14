#!/bin/bash
#SBATCH --job-name=hpml_hw1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%x.out
#SBATCH --mem=32GB
#SBATCH --time=4:00:00

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate env_name
cd /home/

python <your_script>.py