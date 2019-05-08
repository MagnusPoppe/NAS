#!/bin/sh
#SBATCH --partition=EPICALL
#SBATCH --time=7-0:00:00
#SBATCH --nodes=9
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --ntasks-per-node=2
#SBATCH --job-name="EA-NAS"
#SBATCH --output=logs/EA-NAS-CIFAR-AUG-SIM.log
#SBATCH --mail-user=magnuspw@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "-- SLURM INFORMATION --"
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "$SLURM_JOB_NUM_NODES compute nodes with $SLURM_CPUS_ON_NODE processes each"
echo "Total of $SLURM_NTASKS processes with GPUs"
echo ;

# TENSORFLOW 1.10.1 running on CUDA 9.0 setup:
module load GCC/6.4.0-2.28
module load CUDA/9.0.176
module load OpenMPI/2.1.1
module load cuDNN/7
module load Python/3.6.3

# export PATH="/usr/local/cuda-9.0:$PATH"
# export CUDA_HOME="/usr/local/cuda-9.0"
# export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH"
export EA_NAS_UPLOAD_TO_FIREBASE="0"

echo "-- SETTING UP SERVERS --"; echo;
venv/bin/python slurm_nodes.py $SLURM_JOB_NODELIST "./configurations/cifar-10/slurm.json"

echo "-- OUTPUT OF MAIN PROGRAM --"; echo;
mpiexec -n $SLURM_NTASKS venv/bin/python -m mpi4py.futures start_ea_nas.py ./configurations/cifar-10/slurm.json
