#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 -p small-g --gpus-per-node 2 --mem=0 -t 36:00:00 -A project_462000007 --output=slurm_logs/slurm-%x-%j.out
#SBATCH 
#SBATCH

date
rocm-smi
#export NCCL_SOCKET_IFNAME=hsn
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
unset MIOPEN_DISABLE_CACHE
# debugging (noisy)
#export NCCL_DEBUG=INFO
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1 
#export NCCL_DEBUG_SUBSYS=INIT,COLL

#No idea what these do...
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

#export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/scratch/project_462000007/hatanpav/output/datasets/
export HF_METRICS_CACHE=/scratch/project_462000007/hatanpav/output/metrics/
export HF_MODULES_CACHE=/scratch/project_462000007/hatanpav/output/modules/

module --quiet purge

module use /appl/local/csc/soft/ai/modulefiles/
ml pytorch/unstable && echo "using pytorch/unstable"

set -x
#srun mkdir /tmp/python_packages
#srun cp /projappl/project_462000007/hatanpav/minerl/pydreamer/env.zip /tmp/python_packages
#srun unzip /tmp/python_packages/env.zip
#export PYTHONPATH="/tmp/python_packages"

#export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"

#export MLFLOW_TRACKING_URI="https://lumiflow.rahtiapp.fi"
#export MLFLOW_TRACKING_USERNAME="lumi-flow"
#export MLFLOW_TRACKING_PASSWORD="5HIgbIfIb0pKvhNCWuEy"
#export MLFLOW_EXPERIMENT_NAME="minerl_default"

#export MLFLOW_TRACKING_INSECURE_TLS="true"
#export MLFLOW_RUN_NAME=$SLURM_JOB_ID
#export MLFLOW_RESUME_ID=$SLURM_JOB_ID

#srun mkdir /tmp/mlruns
#srun cp -r mlruns /tmp
export MLFLOW_TRACKING_URI="file:///tmp/mlruns"

export MAIN_PY="/projappl/project_462000007/hatanpav/minerl/multi-pydreamer/launch.py --configs defaults minerl"

srun singularity_wrapper exec xvfb-run python3 $MAIN_PY

srun rm -rf /tmp/mlruns
rocm-smi
date