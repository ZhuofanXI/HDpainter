#!/bin/bash -l
#SBATCH --job-name=hdpainter_vae
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/vae_%j.out
#SBATCH --error=logs/vae_%j.err

set -e

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ=/ibex/user/wuj0c/Projects/RNA/HDpainter
CODE=$PROJ/code
DATA=$PROJ/data/NSCLC
CKPT=$PROJ/checkpoints/vae_nsclc

mkdir -p $CODE/logs $CKPT

# ── Environment ───────────────────────────────────────────────────────────────
echo "验证 uv 环境..."
cd $CODE
if ! uv run python -c "import torch" &> /dev/null; then
    echo "错误：uv 环境异常，请先在登录节点运行 uv sync"
    exit 1
fi
echo "✓ uv 环境正常"

# ── Job 信息 ──────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

# ── Training ──────────────────────────────────────────────────────────────────
uv run python scripts/train_vae.py \
    --data_dir    $DATA \
    --ckpt_dir    $CKPT \
    --epochs      100   \
    --batch_size  4     \
    --lr          1e-4  \
    --latent_dim  50    \
    --kl_weight   1e-5  \
    --num_workers 8     \
    --save_every  10

echo "=========================================="
echo "Finished : $(date)"
echo "=========================================="
