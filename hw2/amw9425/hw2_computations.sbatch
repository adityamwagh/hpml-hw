#!/bin/bash

#SBATCH --job-name=hw2_computations
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --gres=gpu
#SBATCH --mail-user=aditya.wagh@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=%x.out

module purge

##############################################################################################################################################################################################
# Question C2 Workers 2 Optimizer SGD Device GPU
##############################################################################################################################################################################################
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c2 -dp data -w 2 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4"

##############################################################################################################################################################################################
## Question C3 Workers 0 Optimizer SGD Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c3 -dp data -w 0 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4"

## Question C3 Workers 4 Optimizer SGD Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c3 -dp data -w 4 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4"

## Question C3 Workers 8 Optimizer SGD Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c3 -dp data -w 8 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4"

## Question C3 Workers 12 Optimizer SGD Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c3 -dp data -w 12 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4"

## Question C3 Workers 16 Optimizer SGD Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c3 -dp data -w 16 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4"

##############################################################################################################################################################################################
## Question C4 Workers 1 Optimizer SGD Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c4 -dp data -w 1 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4"

## Question C4 Workers 4 (optimal) Optimizer SGD Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c4 -dp data -w 4 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4"

##############################################################################################################################################################################################
## Question C5 Workers 4 (optimal) Optimizer SGD Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c5 -dp data -w 4 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4"

## Question C5 Workers 4 (optimal) Optimizer SGD Device CPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c5 -dp data -w 4 -d cpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4"

##############################################################################################################################################################################################
## Question C6 Workers 4 (optimal) Optimizer SGD  Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c6 -dp data -w 4 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4"

## Question C6 Workers 4 (optimal) Optimizer SGD with Nesterov  Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c6 -dp data -w 4 -d gpu -e 5 -o sgdnest -lr 0.1 -m 0.9 -wd 5e-4"

## Question C6 Workers 4 (optimal) Optimizer Adagrad Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c6 -dp data -w 4 -d gpu -e 5 -o adagrad -lr 0.1 -m 0.9 -wd 5e-4"

## Question C6 Workers 4 (optimal) Optimizer Adadelta Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c6 -dp data -w 4 -d gpu -e 5 -o adadelta -lr 0.1 -m 0.9 -wd 5e-4"

## Question C6 Workers 4 (optimal) Optimizer Adam Device GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c6 -dp data -w 4 -d gpu -e 5 -o adam -lr 0.1 -m 0.9 -wd 5e-4"

##############################################################################################################################################################################################
## Question C7 Workers 4 (optimal) Optimizer SGD Device GPU Batchnorm disabled
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab2.py -q c7 -dp data -w 4 -d gpu -e 5 -o sgd -lr 0.1 -m 0.9 -wd 5e-4 -dbn"
