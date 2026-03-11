#!/bin/bash
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=dym
#SBATCH --account=dym-lab
#SBATCH --array=1-5

# 1. Navigate to the working directory
cd ~/PycharmProjects/yon_canon

# 2. Setup your environment
eval "$(conda shell.bash hook)"
conda activate canon

# 3. Capture the Sweep ID passed from the command line
SWEEP_ID=$1
PROJECT="yon_canon"
ENTITY="team_nadav"

if [ -z "$SWEEP_ID" ]; then
    echo "Error: No Sweep ID provided."
    echo "Usage: sbatch --job-name=<name> run_sweep.sbatch <SWEEP_ID>"
    exit 1
fi

mkdir -p logs

# 4. Execute the W&B agent
echo "Starting W&B Agent for Sweep: $SWEEP_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID on node: $SLURM_NODELIST"

wandb agent $ENTITY/$PROJECT/$SWEEP_ID --count 10