#!/bin/bash

#SBATCH --job-name=hw4_v100_computations
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:v100:4
#SBATCH --mail-user=aditya.wagh@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --output=%x.out

module purge

## Training on one GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 1 -e 2 -bs 32"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 1 -e 2 -bs 128"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 1 -e 2 -bs 512"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 1 -e 2 -bs 2048"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 1 -e 2 -bs 8192"

## Training on two GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 2 -e 2 -bs 32"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 2 -e 2 -bs 128"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 2 -e 2 -bs 512"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 2 -e 2 -bs 2048"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 2 -e 2 -bs 8192"

## Training on four GPU
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 4 -e 2 -bs 32"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 4 -e 2 -bs 128"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 4 -e 2 -bs 512"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 4 -e 2 -bs 2048"

singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 4 -e 2 -bs 8192"

# Q4 5 epochs
singularity exec --nv \
--overlay /scratch/amw9425/images/pytorch/my_pytorch.ext3:ro \
/scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python lab4.py -ndev 4 -e 5 -bs 2048"