#!/bin/sh
#SBATCH --partition=EPICALL
#SBATCH --time=00:05:00
#SBATCH --nodes=28
#SBATCH --gres=gpu:2
#SBATCH --mem=3000
#SBATCH --job-name="megatest"
#SBATCH --output=gputest.log
#SBATCH --mail-user=magnuspw@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "-- SLURM INFORMATION --"
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo

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

echo "-- OUTPUT OF MAIN PROGRAM --"; echo;
venv/bin/python slurm_nodes.py "$SLURM_JOB_NODELIST"
