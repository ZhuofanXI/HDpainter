#!/bin/bash -l
#SBATCH --job-name=hdpainter_infer
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err

set -e

PROJ=/ibex/user/wuj0c/Projects/RNA/HDpainter
CODE=$PROJ/code
DATA=$PROJ/data/SVD_CESC
CKPT=$PROJ/checkpoints/ldm_cesc/latest.pt
OUT=$PROJ/outputs/infer

mkdir -p $CODE/logs $OUT

echo "验证 uv 环境..."
cd $CODE
if ! uv run python -c "import torch" &> /dev/null; then
    echo "错误：uv 环境异常"
    exit 1
fi
echo "✓ uv 环境正常"

echo "=========================================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

uv run python -u scripts/infer.py \
    --ckpt       $CKPT  \
    --data_dir   $DATA  \
    --out_dir    $OUT   \
    --n_tiles    10     \
    --n_vis      3      \
    --ddim_steps 50

echo "=========================================="
echo "Finished : $(date)"
echo "=========================================="
