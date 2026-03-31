#!/bin/bash -l
#SBATCH --job-name=hdpainter_ldm
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/ldm_%j.out
#SBATCH --error=logs/ldm_%j.err

set -e

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ=/ibex/user/wuj0c/Projects/RNA/HDpainter
CODE=$PROJ/code
DATA=$PROJ/data/SVD_CESC
CKPT=$PROJ/checkpoints/ldm_cesc

mkdir -p $CODE/logs $CKPT

# ── Environment check ─────────────────────────────────────────────────────────
echo "验证 uv 环境..."
cd $CODE
if ! uv run python -c "import torch" &> /dev/null; then
    echo "错误：uv 环境异常，请先在登录节点运行 uv sync"
    exit 1
fi
echo "✓ uv 环境正常"

# ── Job info ──────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURMD_NODENAME"
echo "Started : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================="

# ── Train ─────────────────────────────────────────────────────────────────────
uv run python -u scripts/train_ldm.py \
    --data_dir      $DATA \
    --ckpt_dir      $CKPT \
    --epochs        200   \
    --batch_size    2     \
    --num_workers   8     \
    --lr            1e-4  \
    --save_every    10    \
    --T             1000  \
    --t_thresh      200   \
    --base_ch       64    \
    --ch_mult       1 2 4 8 \
    --num_res_blocks 2

echo "=========================================="
echo "Finished : $(date)"
echo "=========================================="
