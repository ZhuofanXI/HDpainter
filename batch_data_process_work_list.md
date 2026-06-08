# HDpainter Batch Preprocess Work List

更新时间：2026-06-04

本文档是服务器迁移后的批量预处理接手清单。COAD 已作为参考样本跑通；后续只处理 PRAD 和 NSCLC，顺序固定为：

```text
PRAD -> NSCLC
```

不要处理 LIHC。

## 远端连接与目录

```bash
ssh -p 21441 root@connect.bjb1.seetacloud.com
```

标准远端环境：

```text
project root: /root/autodl-tmp/HDpainter1
raw data:     /root/autodl-tmp/HDpainter1/raw_data
output root:  /root/autodl-tmp/OV/batch_preprocess
log root:     /root/autodl-tmp/OV/batch_preprocess_logs
python:       /root/miniconda3/envs/czf/bin/python
```

最终目标文件：

```text
/root/autodl-tmp/OV/batch_preprocess/PRAD/regularize_train_tiles_degraded_PRAD_nmf48.instchunk512_train.maskrefined.h5
/root/autodl-tmp/OV/batch_preprocess/NSCLC/regularize_train_tiles_degraded_NSCLC_nmf48.instchunk512_train.maskrefined.h5
```

## 开始前必须检查

避免重复启动：

```bash
ps -eo pid,ppid,stat,pcpu,pmem,rss,etime,cmd | egrep 'batch_process_raw_data.py|preprocess.py train-full' | grep -v egrep
```

确认主链文件存在：

```bash
PROJECT=/root/autodl-tmp/HDpainter1
ls -lh \
  "$PROJECT/preprocess/batch_process_raw_data.py" \
  "$PROJECT/preprocess/preprocess.py" \
  "$PROJECT/preprocess/utils.py" \
  "$PROJECT/preprocess/old_code/regularize_cell_masks.py" \
  "$PROJECT/preprocess/old_code/mask_refine.py"
```

远端语法检查：

```bash
PY=/root/miniconda3/envs/czf/bin/python
PROJECT=/root/autodl-tmp/HDpainter1

"$PY" -m py_compile \
  "$PROJECT/preprocess/batch_process_raw_data.py" \
  "$PROJECT/preprocess/preprocess.py" \
  "$PROJECT/preprocess/utils.py" \
  "$PROJECT/preprocess/old_code/regularize_cell_masks.py" \
  "$PROJECT/preprocess/old_code/mask_refine.py"
```

资源检查：

```bash
cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes
free -h
df -h /root/autodl-tmp
```

62GB 服务器默认参数：

```text
--memory-limit-gb 62
--min-free-disk-gb 120
--regularize-num-workers 8
--mask-refine-num-workers 8
```

## 正确标准流程

当前标准不是 HD expanded pseudo-cell NMF，也不是全 bin NMF。

正确逻辑：

1. 用 Xenium transcripts、cell boundaries、nucleus boundaries 构造 union-cell pseudo-cell。
2. 对 pseudo-cell 做 QC：`total_counts > 1`、`n_genes_by_counts > 3`、gene-level `expressing_cells > 3`。
3. 在 QC 后 Xenium union-cell pseudo-cell 上拟合 NMF：`--nmf-fit-source xenium-union-cell`、`--nmf-components 48`。
4. HD reference 只用于对齐和 foreground/background degrade 统计。
5. 如果 HD reference 缺少 `labels_he_expanded`，流程用 `labels_he` 或 `stardist_id/cellpose_id` 生成；必要时通过 `--reference-image-path` 自动 StarDist。
6. 用 gamma-poisson degrade 后的 synthetic Xenium HD bins 构建训练 H5。
7. 下游依次生成 regularize、direction targets、microenv、instance chunks、mask refine 产物。

禁止回退到：

```text
全 bin NMF
HD expanded pseudo-cell NMF
fixed-basis MU projection fallback
aligned_cell 与 NMF source cell_id 不一致后的二次投影
```

## 原始数据清单

PRAD：

```text
/root/autodl-tmp/HDpainter1/raw_data/HD/PRAD/Visium_HD_Human_Prostate_Cancer_FFPE_binned_outputs.tar.gz
/root/autodl-tmp/HDpainter1/raw_data/HD/PRAD/Visium_HD_Human_Prostate_Cancer_FFPE_tissue_image.tif
/root/autodl-tmp/HDpainter1/raw_data/Xenium/PRAD/Xenium_Prime_Human_Prostate_FFPE_outs.zip
```

NSCLC：

```text
/root/autodl-tmp/HDpainter1/raw_data/HD/NSCLC/Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_binned_outputs.tar.gz
/root/autodl-tmp/HDpainter1/raw_data/HD/NSCLC/Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_tissue_image.btf
/root/autodl-tmp/HDpainter1/raw_data/Xenium/NSCLC/Xenium_V1_Human_Lung_Cancer_FFPE_outs.zip
```

reference h5ad 标准位置：

```text
/root/autodl-tmp/HDpainter1/raw_data/HD/<SAMPLE>/reference_hd_square_002um.h5ad
```

每个样本启动前必须检查 reference 是否存在。不存在时不要加 `--skip-reference`；存在时才加 `--skip-reference`。

## 运行命令

PRAD reference 不存在时：

```bash
PY=/root/miniconda3/envs/czf/bin/python
PROJECT=/root/autodl-tmp/HDpainter1

"$PY" "$PROJECT/preprocess/batch_process_raw_data.py" \
  --samples PRAD \
  --skip-extract \
  --memory-limit-gb 62 \
  --min-free-disk-gb 120 \
  --regularize-num-workers 8 \
  --mask-refine-num-workers 8
```

PRAD reference 已存在时：

```bash
"$PY" "$PROJECT/preprocess/batch_process_raw_data.py" \
  --samples PRAD \
  --skip-extract \
  --skip-reference \
  --memory-limit-gb 62 \
  --min-free-disk-gb 120 \
  --regularize-num-workers 8 \
  --mask-refine-num-workers 8
```

PRAD 最终 H5 验证成功后，再启动 NSCLC。

NSCLC reference 不存在时：

```bash
"$PY" "$PROJECT/preprocess/batch_process_raw_data.py" \
  --samples NSCLC \
  --skip-extract \
  --memory-limit-gb 62 \
  --min-free-disk-gb 120 \
  --regularize-num-workers 8 \
  --mask-refine-num-workers 8
```

NSCLC reference 已存在时：

```bash
"$PY" "$PROJECT/preprocess/batch_process_raw_data.py" \
  --samples NSCLC \
  --skip-extract \
  --skip-reference \
  --memory-limit-gb 62 \
  --min-free-disk-gb 120 \
  --regularize-num-workers 8 \
  --mask-refine-num-workers 8
```

每次只运行一个样本。

## 断点续跑

如果失败，先读取对应日志：

```bash
ls -lt /root/autodl-tmp/OV/batch_preprocess_logs | head -30
tail -160 /root/autodl-tmp/OV/batch_preprocess_logs/<SAMPLE>_preprocess_<STEP>.log
```

优先从失败步骤继续：

```bash
"$PY" "$PROJECT/preprocess/batch_process_raw_data.py" \
  --samples <SAMPLE> \
  --skip-extract \
  --skip-reference \
  --start-from <FAILED_STEP> \
  --memory-limit-gb 62 \
  --min-free-disk-gb 120 \
  --regularize-num-workers 8 \
  --mask-refine-num-workers 8
```

如果 reference 不存在，续跑命令也不要加 `--skip-reference`。

## 输出验证

PRAD/NSCLC 都必须验证最终 H5 能打开，并包含训练所需 datasets：

```bash
for SAMPLE in PRAD NSCLC; do
  FINAL=/root/autodl-tmp/OV/batch_preprocess/$SAMPLE/regularize_train_tiles_degraded_${SAMPLE}_nmf48.instchunk512_train.maskrefined.h5
  echo "checking $FINAL"
  FINAL="$FINAL" /root/miniconda3/envs/czf/bin/python - <<'PY'
import os
from pathlib import Path
import h5py

p = Path(os.environ["FINAL"])
assert p.exists() and p.stat().st_size > 0, p
required = [
    "train_chunk_tile_offsets",
    "train_chunk_instance_offsets",
    "train_tile_input_pool",
    "train_instance_mask_targets_pool",
    "val_chunk_tile_offsets",
    "val_chunk_instance_offsets",
    "val_tile_input_pool",
    "val_instance_mask_targets_pool",
]
with h5py.File(p, "r") as f:
    missing = [name for name in required if name not in f]
    assert not missing, missing
    print("ok", p.name)
    print("dataset_format", f.attrs.get("dataset_format"))
    print("mask_refine_version", f.attrs.get("mask_refine_version"))
    print("n_train_chunks", f.attrs.get("n_train_chunks"))
    print("n_val_chunks", f.attrs.get("n_val_chunks"))
    for name in required:
        print(name, f[name].shape)
PY
done
```

## COAD 参考样本记录

COAD 已证明标准流程可行，关键经验如下：

```text
NMF source: nmf_source_cells.h5ad shape = (403811, 4821)
degraded.h5ad shape = (11868512, 4821)
NMF elapsed: about 20 min
NMF peak RSS: about 20GB
build_h5 exported tiles: 310
instance_chunks train/val chunks: 617 / 70
instance_chunks train/val instances: 315476 / 35335
```

经验结论：

```text
NMF 不应超过小时级；若长时间不结束，优先检查是否误入全 bin 或错误投影路径。
regularize 和 mask_refine 默认用 8 workers。
mask_refine.py 必须存在并同步到远端。
reference 缺失时不能使用 --skip-reference。
```
