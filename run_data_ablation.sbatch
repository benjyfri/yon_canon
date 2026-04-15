#!/bin/bash
#SBATCH --job-name=data_ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=newton1,ran-mashawsha    # <--- THE BAN LIST
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-15

# 1. Navigate to the working directory and setup environment
cd ~/PycharmProjects/yon_canon
eval "$(conda shell.bash hook)"
conda activate canon

mkdir -p logs

# 2. Define the exact grid for the Ablation Study
STRIDES=(1 2 4 8 16) # 1=100%, 2=50%, 4=25%, 8=12.5%, 16=6.25% of data
ORDERINGS=("ply" "lex" "hilbert")

# Calculate indices based on the SLURM_ARRAY_TASK_ID (which goes 1 to 15)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
STRIDE_IDX=$((IDX % 5))
ORD_IDX=$((IDX / 5))

CURRENT_STRIDE=${STRIDES[$STRIDE_IDX]}
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
EXP_NAME="ablation_${CURRENT_ORD}_stride_${CURRENT_STRIDE}"

echo "================================================================="
echo "Starting Job ID: $SLURM_ARRAY_TASK_ID"
echo "Ordering: $CURRENT_ORD | Stride: $CURRENT_STRIDE (1/${CURRENT_STRIDE} of data)"
echo "Hyperparameters -> LR: $LR, Dropout: $DROPOUT, Weight Decay: $WD"
echo "================================================================="

# 4. Execute the training script
# Note: --apply_jitter and --apply_scale are OMITTED as requested.
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
    --dataset_stride $CURRENT_STRIDE \
    --num_points 1024