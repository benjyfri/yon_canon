#!/bin/bash
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton1,ran-mashawsha
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-5

set -euo pipefail

cd ~/PycharmProjects/yon_canon

eval "$(conda shell.bash hook)"
conda activate canon

SWEEP_ID="${1:-}"
PROJECT="yon_canon_new"
ENTITY="team_nadav"

if [[ -z "$SWEEP_ID" ]]; then
    echo "Error: No Sweep ID provided."
    echo "Usage: sbatch --job-name=<name> run_sweep.sbatch <SWEEP_ID>"
    exit 1
fi

mkdir -p logs

echo "Starting W&B agent for sweep: $SWEEP_ID"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node list: ${SLURM_NODELIST}"

wandb agent "${ENTITY}/${PROJECT}/${SWEEP_ID}" --count 10