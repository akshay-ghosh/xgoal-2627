#!/bin/bash
#SBATCH --time=01:30:00 # time is hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G # multiply by cpus-per-task for total
#SBATCH --output=slurm_logs/xgoal_slurm-%j.out
#SBATCH --account=def-aghosh

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
pip install --no-index torch

# launch python file
python nn_goals.py