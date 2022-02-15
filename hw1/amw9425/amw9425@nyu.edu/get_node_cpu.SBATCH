#!/bin/bash
#SBATCH --job-name=hpml_hw1_cpu
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

# run dp1.c
gcc -O3 -Wall -o dp1.out dp1.c 
./dp1.out 1000000 1000

# run dp4.py
python dp4.py 1000000 1000