#!/bin/sh
#SBATCH --partition=EPIC2
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:V100:3
#SBATCH --mem=12000
# Memory per node specification is in MB. It is optional. 
# The default limit is 3000MB per core.
#SBATCH --job-name="TESTING GPU LAYOUT"
#SBATCH --output=test.out
#SBATCH --mail-user=magnuspw@stud.ntnu.no
#SBATCH --mail-type=ALL
 
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
 
# TENSORFLOW 1.10.1 running on CUDA 9.0 setup:
module load GCC/6.4.0-2.28
module load CUDA/9.0.176
module load OpenMPI/2.1.1
module load cuDNN/7
module load Python/3.6.3

export PATH="/usr/local/cuda-9.0:$PATH"
export CUDA_HOME="/usr/local/cuda-9.0"
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH"
export EA_NAS_UPLOAD_TO_FIREBASE="0"

cd /home/magnuspw/EPIC_server_testing
py363/bin/python -c "import subprocess; print(subprocess.check_output(['nvidia-smi', '-L']))"