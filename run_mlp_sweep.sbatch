#!/bin/bash
#SBATCH --output=logs/mlp_sweep_%A_%a.out
#SBATCH --error=logs/mlp_sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                  # Requesting any 1 GPU (cheaper than L40/A40)
#SBATCH --mem=16G                     # Lower memory footprint
#SBATCH --time=12:00:00               # 12 hours is plenty for MLP
#SBATCH --partition=dym               # Use the default or a "low" partition if available
#SBATCH --account=dym-lab
#SBATCH --array=1-3                   # 3 parallel workers for each sweep

# 1. Navigate to the working directory
cd ~/PycharmProjects/yon_canon

# 2. Setup your environment
eval "$(conda shell.bash hook)"
conda activate canon

# 3. Capture the Sweep ID
SWEEP_ID=$1
PROJECT="yon_canon"
ENTITY="team_nadav"

if [ -z "$SWEEP_ID" ]; then
    echo "Error: No Sweep ID provided."
    exit 1
fi

mkdir -p logs

# 4. Execute the W&B agent
echo "Starting MLP Agent for Sweep: $SWEEP_ID"
wandb agent $ENTITY/$PROJECT/$SWEEP_ID --count 15