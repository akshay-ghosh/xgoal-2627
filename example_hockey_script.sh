#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --mem-per-cpu=1.5G      # increase as needed
#SBATCH --time=1:00:00

#SBATCH --time=01:00:00
#SBATCH --cpus-per-node=2
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=smith

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt

python ...  # your python script here

'''
additional sbatch options:

#!/bin/bash
#SBATCH --account=def-smith
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00

https://docs.alliancecan.ca/wiki/Python#Creating_virtual_environments_inside_of_your_jobs
'''

'''
result of
gtime -v /Users/akshayghosh/acenet_py/deep_learning/.venv/bin/python3 nn_xgoals.py

	Command being timed: "/Users/akshayghosh/acenet_py/deep_learning/.venv/bin/python3 nn_xgoals.py"
	User time (seconds): 8897.97
	System time (seconds): 3904.05
	Percent of CPU this job got: 361%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 59:05.59
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 7711732
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 2066
	Minor (reclaiming a frame) page faults: 1128086797
	Voluntary context switches: 10428
	Involuntary context switches: 11564347
	Swaps: 0
	File system inputs: 0
	File system outputs: 0
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
'''