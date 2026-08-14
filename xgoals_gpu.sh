#!/bin/bash
#SBATCH --time=01:30:00 # time is hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G # multiply by cpus-per-task for total
#SBATCH --output=slurm_logs/xgoal_slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --account=def-aghosh

## ^^ EDIT SBATCH COMMANDS TO USE GPU

# load modules
module load python/3.13.2 scipy-stack

# create and activate virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# install packages
pip install --no-index --upgrade pip
pip install --no-index numpy
pip install --no-index matplotlib
pip install --no-index pandas
pip install --no-index seaborn
pip install --no-index sklearn
pip install --no-index scipy
pip install --no-index torch # this is confirmed working on narval

# launch python file
python nn_xgoals_gpu.py

'''
salloc for interactive job
You can start an interactive session on a compute node with salloc. In the following example we request one task, 
which corresponds to one CPU cores and 3 GB of memory, for an hour: 

salloc --time=1:0:0 --mem-per-cpu=1G --ntasks=1 --account=def-aghosh
request gpu: --gpus=a100_1g.5gb:1

try this:

salloc --time=1:0:0 --mem-per-cpu=1G --ntasks=1 --gpus=a100_1g.5gb:1 --account=def-aghosh

when i run it with the simple gpu test script:

device is cuda
X_train on gpu: True
number of gpus available = 1
name of gpu device = NVIDIA A100-SXM4-40GB MIG 1g.5gb
'''
