#!/bin/bash

#SBATCH --job-name=hw2_computations
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu
#SBATCH --mail-user=aditya.wagh@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=%x.out

module purge

## Question C7 Workers 4 (optimal) Optimizer SGD Device GPU Batchnorm disabled
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c7 -dp ./data -w 4 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4 -dbn"
