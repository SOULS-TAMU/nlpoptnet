#!/bin/bash
#SBATCH --job-name=nlpopt-exprs-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=360G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/user/%u/NLPOptNet/logs/%x-%j.out
#SBATCH --error=/scratch/user/%u/NLPOptNet/logs/%x-%j.err

set -euo pipefail

module purge
module use /scratch/user/$USER/modulefiles
module load nlpopt

REPO_ROOT="${NLP_OPT_ROOT:-/scratch/user/$USER/NLPOptNet}"

mkdir -p "$REPO_ROOT/logs"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED=1

# GPU / JAX setup
export JAX_PLATFORMS=cuda
unset CUDA_VISIBLE_DEVICES

# Threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Avoid grabbing all GPU memory up front
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "========================================"
echo "Job ID      : ${SLURM_JOB_ID:-unknown}"
echo "Job Name    : ${SLURM_JOB_NAME:-unknown}"
echo "Partition   : ${SLURM_JOB_PARTITION:-unknown}"
echo "Node        : $(hostname)"
echo "Start Time  : $(date)"
echo "Repo Root   : $REPO_ROOT"
echo "CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "GPUs        : ${SLURM_GPUS:-1}"
echo "Memory      : ${SLURM_MEM_PER_NODE:-360G}"
echo "========================================"

echo
echo "===== GPU info ====="
nvidia-smi || true
echo "===================="
echo

python - <<'PY'
import jax
import torch

print("JAX devices:", jax.devices())
print("JAX default backend:", jax.default_backend())
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Torch current device:", torch.cuda.current_device())
    print("Torch device name:", torch.cuda.get_device_name(0))
PY

##### Job 1 #####
srun python main.py --type qp --action compare --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
srun python main.py --type qcqp --action compare --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
srun python main.py --type nlp --action compare --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
srun python main.py --type nonconvx --action compare --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8

##### Job 2 #####
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.6
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.4
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.2

srun python main.py --type qp --action compare_dc3 --p 25 --n 100 --me 50 --mi 50 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 75 --n 100 --me 50 --mi 50 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 100 --n 100 --me 50 --mi 50 --train_frac 0.8

##### Job 3 #####
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 10 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 30 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 50 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 70 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --me 50 --mi 90 --train_frac 0.8

srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --mi 50 --me 10 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --mi 50 --me 30 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --mi 50 --me 50 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --mi 50 --me 70 --train_frac 0.8
srun python main.py --type qp --action compare_dc3 --p 50 --n 100 --mi 50 --me 90 --train_frac 0.8

##### Job 4 #####
# srun python main.py --type qcqp --action compare_dc3 --p 100 --n 1000 --mi 500 --me 500 --train_frac 0.8

echo "========================================"
echo "End Time    : $(date)"
echo "========================================"