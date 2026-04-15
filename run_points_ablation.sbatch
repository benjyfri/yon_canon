#!/bin/bash
#SBATCH --job-name=pts_ablation
#SBATCH --output=logs/pts_ablation_%A_%a.out
#SBATCH --error=logs/pts_ablation_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton1,ran-mashawsha    # <--- THE BAN LIST
#SBATCH --mem=32G
#SBATCH --time=04:00:00       # Short time limit to enable backfilling
#SBATCH --array=1-15

# (Optional: If IT requires you to use the dym partition, uncomment the two lines below)
# #SBATCH --partition=dym
# #SBATCH --account=dym-lab

# 1. Navigate to the working directory and setup environment
cd ~/PycharmProjects/yon_canon
eval "$(conda shell.bash hook)"
conda activate canon

mkdir -p logs

# 2. Define the exact grid for the Ablation Study
POINTS=(1024 512 256 128 64)
ORDERINGS=("ply" "lex" "hilbert")

# Calculate indices based on the SLURM_ARRAY_TASK_ID (which goes 1 to 15)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
PTS_IDX=$((IDX % 5))
ORD_IDX=$((IDX / 5))

CURRENT_PTS=${POINTS[$PTS_IDX]}
CURRENT_ORD=${ORDERINGS[$ORD_IDX]}

# 3. Load the best hyperparameters for the selected ordering
if [ "$CURRENT_ORD" == "ply" ]; then
    LR=0.0014988353560536768
    DROPOUT=0.1
    WD=0.05
    F_SCALE=1.0
    LS=0.1
elif [ "$CURRENT_ORD" == "lex" ]; then
    LR=0.0006407203395820919
    DROPOUT=0.1
    WD=0.05
    F_SCALE=0.1
    LS=0.2
elif [ "$CURRENT_ORD" == "hilbert" ]; then
    LR=0.0009614328324244756
    DROPOUT=0.3
    WD=0.01
    F_SCALE=0.1
    LS=0.2
fi

# Set a dynamic experiment name for W&B
EXP_NAME="ablation_${CURRENT_ORD}_pts_${CURRENT_PTS}"

echo "================================================================="
echo "Starting Job ID: $SLURM_ARRAY_TASK_ID"
echo "Running on Node: $SLURM_NODELIST"
echo "Assigned GPU(s): $CUDA_VISIBLE_DEVICES"
echo "Ordering: $CURRENT_ORD | Points: $CURRENT_PTS (Full Train Set)"
echo "Hyperparameters -> LR: $LR, Dropout: $DROPOUT, Weight Decay: $WD"
echo "================================================================="

# Print GPU state to log before training
nvidia-smi

# 4. Execute the training script
# Note: dataset_stride is locked to 1 (full dataset) and augs are omitted.
python train.py \
    --exp_name "$EXP_NAME" \
    --dataset "modelnet40" \
    --model "global_mlp" \
    --ordering "$CURRENT_ORD" \
    --epochs 100 \
    --batch_size 256 \
    --test_batch_size 16 \
    --optimizer "adamw" \
    --lr $LR \
    --dropout $DROPOUT \
    --weight_decay $WD \
    --num_bands 3 \
    --fourier_scale $F_SCALE \
    --label_smoothing $LS \
    --dataset_stride 1 \
    --num_points $CURRENT_PTS