#!/bin/bash
#SBATCH --job-name=hpml_hw1_cpu
#SBATCH --output=slurm_%x_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=aditya.wagh@nyu.edu
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

## load modules
module purge
module load intel/19.1.2
module load python/intel/3.8.6

## change to working dir
cd /scratch/amw9425/hpml_hw/hw1/amw9425/amw9425@nyu.edu

## execute code for dp1
gcc -O3 -Wall -I /share/apps/intel/19.1.2/mkl/include/ -L /share/apps/intel/19.1.2/mkl/lib/intel64/ -o dp1 dp1.c -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
./dp1 1000000 1000
./dp1 300000000 20

## execute code for dp1
gcc -O3 -Wall -I /share/apps/intel/19.1.2/mkl/include/ -L /share/apps/intel/19.1.2/mkl/lib/intel64/ -o dp2 dp2.c -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
./dp2 1000000 1000
./dp2 300000000 20

## execute code for dp3
gcc -O3 -Wall -I /share/apps/intel/19.1.2/mkl/include/ -L /share/apps/intel/19.1.2/mkl/lib/intel64/ -o dp3 dp3.c -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
./dp3 1000000 1000
./dp3 300000000 20

## execute code for dp4
python dp4.py 1000000 1000
python dp4.py 300000000 20

## execute code for dp5
python dp5.py 1000000 1000
python dp5.py 300000000 20
