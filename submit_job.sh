#!/bin/bash
#SBATCH --job-name=rhea-soil
#SBATCH --output=rhea_%j.out
#SBATCH --error=rhea_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_normal_q

# Load modules (adjust to your cluster)
module load Python/3.11 2>/dev/null || true
module load CUDA 2>/dev/null || true

# Install deps if needed
pip install --user lightgbm xgboost catboost scikit-learn pandas numpy scipy 2>/dev/null

# Run
cd $SLURM_SUBMIT_DIR
python -u run_gpu.py
