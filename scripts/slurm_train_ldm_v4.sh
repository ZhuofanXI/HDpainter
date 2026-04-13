#!/bin/bash -l
#SBATCH --job-name=hdpainter_v4
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=logs/ldm_v4_%j.out
#SBATCH --error=logs/ldm_v4_%j.err

set -e

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ=/ibex/user/wuj0c/Projects/RNA/HDpainter
CODE=$PROJ/code
SRC_DATA=$PROJ/data          # 包含 SVD_CESC/ SVD_OV/ 子目录
DST_DATA=$PROJ/data/HD_dataset_nuc10
CKPT=$PROJ/checkpoints/ldm_v4_cesc_ov_nuc10

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

# ── Step 1: Offline preprocessing (skip if dst already exists and non-empty) ──
if [ -z "$(ls -A $DST_DATA 2>/dev/null)" ]; then
    echo "--- 预处理开始 (min_nuc=10) ---"
    uv run python -u scripts/preprocess.py \
        --src_dir   $SRC_DATA \
        --dst_dir   $DST_DATA \
        --min_nuc   10        \
        --patch_size 128      \
        --overlap   16
    echo "--- 预处理完成 ---"
else
    echo "--- 检测到已有预处理数据，跳过预处理 ---"
fi

# ── Step 2: Train ─────────────────────────────────────────────────────────────
echo "--- 训练开始 ---"
uv run python -u scripts/train_ldm.py \
    --data_dir      $DST_DATA \
    --ckpt_dir      $CKPT     \
    --epochs        200       \
    --batch_size    2         \
    --num_workers   8         \
    --lr            1e-4      \
    --save_every    10        \
    --T             1000      \
    --t_thresh      200       \
    --base_ch       64        \
    --ch_mult       1 2 4 8   \
    --num_res_blocks 2

echo "=========================================="
echo "Finished : $(date)"
echo "=========================================="
