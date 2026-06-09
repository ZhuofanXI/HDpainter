# HDpainter

HDpainter is a cross-platform pipeline for Visium HD cell segmentation and cell-level signal enhancement. It uses high-confidence Xenium cell/nucleus annotations to synthesize Visium HD-like training data, trains a supervised segmentation model on the synthetic HD representation, applies the model to real Visium HD data, and then performs graph neural network post-processing to improve cell-level gene signals.

## Core Idea

HDpainter is built around one practical observation: Xenium provides transcript-level coordinates and 10x cell/nucleus segmentation, while Visium HD provides dense 2 um bin-level spatial transcriptomics but lacks reliable cell-level masks. HDpainter transfers the segmentation supervision from Xenium to Visium HD by creating synthetic HD-style training data from Xenium.

The training data construction follows these steps:

1. Xenium transcripts, cell boundaries, and nucleus boundaries are read together.
2. Cell and nucleus masks are merged into union-cell instances. These instances are used to aggregate Xenium transcripts into pseudo-single-cell profiles.
3. Low-quality pseudo-cells are filtered before NMF fitting. The current default QC is `total_counts > 1`, `n_genes_by_counts > 3`, and gene-level `expressing_cells > 3`.
4. NMF is fitted on the QC-passed Xenium union-cell pseudo-single-cell matrix. The default latent dimension is `48`.
5. The fitted NMF basis is applied to synthetic Xenium HD bins and real Visium HD bins, so the segmentation model sees the same latent feature space during training and inference.
6. A real Visium HD reference sample is used to estimate foreground/background expression statistics. HD H&E-based StarDist/bin2cell labels, especially `labels_he_expanded`, separate cell and non-cell regions.
7. Synthetic Xenium HD bins are degraded with gene-wise foreground/background gamma-poisson statistics estimated from real HD data. This makes the synthetic supervision closer to real Visium HD noise and sparsity.
8. The final training H5 stores tile-level features, instance masks, local context, direction targets, microenvironment features, train/validation chunks, and mask-refined targets.

During inference, real Visium HD data are first converted into the same bin-level feature representation. H&E-based nucleus segmentation can be generated automatically when the required nucleus labels are missing. The trained HDpainter model predicts cell masks from nuclei/seeds and local HD transcriptomic context, writes predicted cell IDs back to the bin-level AnnData, and aggregates bins into segmentation-derived pseudo-single-cell profiles.

After segmentation, `post_process.py` and `signal_process.py` provide the cell-segmentation-based graph signal-processing stage. `post_process.py` validates and prepares the cell-level AnnData. `signal_process.py` builds local spatial and expression graphs on segmented cells, trains a relational graph autoencoder, saves the cell embedding in `obsm["GNN"]`, and saves full-gene reconstructed expression in `layers["GNN_ReX"]`.

Supplementary note: HDpainter's data preprocessing workflow, segmentation model, and post-processing code were manually designed and implemented. Codex was mainly used to assist with batch data processing, result visualization, code-structure optimization, and automated testing during development. All code has been manually reviewed. If you use Codex or a similar coding agent with this project, you can ask it to read the corresponding markdown files and run the documented tests directly.

## Repository Layout

```text
HDpainter/
├── preprocess/
│   ├── batch_process_raw_data.py    # Batch raw-data driver for Xenium + Visium HD preprocessing
│   ├── preprocess.py                # Main preprocessing stages
│   ├── utils.py                     # Preprocessing helpers
│   └── visual_sys_hd.py             # Training-H5 visualization and statistics
├── model/
│   ├── dataset.py                   # H5 dataset reader and tile/instance batching
│   ├── model.py                     # HDpainter segmentation model
│   └── train.py                     # Supervised training entry point
├── inference/
│   ├── nucleus_segment.py           # H&E-based nucleus segmentation helper
│   ├── infer_hd.py                  # One-command real-HD inference pipeline
│   ├── infer_utils.py               # Inference-stage helpers
│   ├── post_process.py              # Cell-level AnnData QC and graph-input preparation
│   ├── signal_process.py            # Graph signal reconstruction
│   ├── post_data_check.py           # Cell-level QC distribution plots
│   ├── compare_data.py              # Compare bin2cell/direct/GNN signal quality
│   └── evaluate_signal_quality.py   # Held-out reconstruction and marker spatial quality
├── validation/
│   ├── bin2cell_validation_baseline.py
│   └── summarize_validation_suite.py
└── batch_data_process_work_list.md  # Current batch preprocessing operation checklist
```

Only Python scripts and the batch preprocessing markdown are intended to be versioned here. Large outputs such as `model/runs/`, `.h5`, `.h5ad`, `.pt`, raw data, and paper/reference folders should remain outside git.

## Required Data

For each training sample, prepare one matched Xenium dataset and one Visium HD reference dataset.

Xenium input should include:

- A Xenium `outs.zip` or extracted `outs/` directory.
- Transcript coordinates.
- 10x cell boundary polygons.
- 10x nucleus boundary polygons.

Visium HD input should include:

- 10x Visium HD `binned_outputs.tar.gz` or extracted binned outputs.
- A matched H&E tissue image, usually `.tif` or `.btf`.
- Optional precomputed `reference_hd_square_002um.h5ad`.

For real-HD inference, prepare:

- A bin-level Visium HD `.h5ad`.
- A trained HDpainter checkpoint.
- The NMF basis/reference AnnData used by the trained model.
- A full-resolution H&E image if nucleus labels need to be generated.

## Recommended Directory Structure

The server workflow used during development assumes this layout:

```text
/root/autodl-tmp/HDpainter1/
├── raw_data/
│   ├── Xenium/
│   │   ├── COAD/
│   │   │   └── <Xenium outs.zip or extracted outs>
│   │   ├── PRAD/
│   │   └── NSCLC/
│   └── HD/
│       ├── COAD/
│       │   ├── <Visium HD binned_outputs.tar.gz>
│       │   ├── <H&E tissue image>
│       │   └── reference_hd_square_002um.h5ad
│       ├── PRAD/
│       └── NSCLC/
├── preprocess/
├── model/
├── inference/
└── validation/
```

The default batch output root is:

```text
/root/autodl-tmp/OV/batch_preprocess/<SAMPLE>/
```

The expected final training H5 is:

```text
/root/autodl-tmp/OV/batch_preprocess/<SAMPLE>/regularize_train_tiles_degraded_<SAMPLE>_nmf48.instchunk512_train.maskrefined.h5
```

## Environment

The project is designed to run on a Linux GPU server. The development environment used:

```text
python: /root/miniconda3/envs/czf/bin/python
project root: /root/autodl-tmp/HDpainter1
```

Core dependencies include `numpy`, `scipy`, `pandas`, `anndata`, `scanpy`, `h5py`, `scikit-learn`, `torch`, `torchvision`, `torch_geometric`, `bin2cell`, `stardist`, `tifffile`, and plotting libraries. The GNN signal-processing stage requires compatible `torch` and `torch_geometric` builds.

For subsequent preprocessing runs, the observed memory pressure was moderate. A 62 GB memory server is expected to be sufficient for the current workflow, with 8 workers for regularization and mask refinement:

```text
--memory-limit-gb 62
--min-free-disk-gb 120
--regularize-num-workers 8
--mask-refine-num-workers 8
```

## 1. Batch Preprocess Xenium + Visium HD

Before running, check that no old preprocessing job is still active:

```bash
ps -eo pid,ppid,stat,pcpu,pmem,rss,etime,cmd | \
  egrep 'batch_process_raw_data.py|preprocess.py train-full' | grep -v egrep
```

Run a syntax check:

```bash
PY=/root/miniconda3/envs/czf/bin/python
PROJECT=/root/autodl-tmp/HDpainter1

"$PY" -m py_compile \
  "$PROJECT/preprocess/batch_process_raw_data.py" \
  "$PROJECT/preprocess/preprocess.py" \
  "$PROJECT/preprocess/utils.py"
```

Run one sample at a time. If `raw_data/HD/<SAMPLE>/reference_hd_square_002um.h5ad` already exists, add `--skip-reference`. Otherwise do not add it.

Example without an existing reference:

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

Example with an existing reference:

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

The current standard sample order is:

```text
COAD -> PRAD -> NSCLC
```

`LIHC` is intentionally excluded until a suitable HD reference is available.

## 2. Validate the Preprocessed Training H5

After preprocessing finishes, check that the final H5 exists and contains the expected train/validation chunk datasets:

```bash
FINAL=/root/autodl-tmp/OV/batch_preprocess/PRAD/regularize_train_tiles_degraded_PRAD_nmf48.instchunk512_train.maskrefined.h5

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
    for name in required:
        print(name, f[name].shape)
PY
```

Optional visualization:

```bash
/root/miniconda3/envs/czf/bin/python \
  /root/autodl-tmp/HDpainter1/preprocess/visual_sys_hd.py \
  --input-h5 /root/autodl-tmp/OV/batch_preprocess/COAD/regularize_train_tiles_degraded_COAD_nmf48.instchunk512_train.maskrefined.h5 \
  --output-dir /root/autodl-tmp/HDpainter1/preprocess/visiualization/COAD
```

## 3. Train HDpainter

Train on a final mask-refined training H5:

```bash
cd /root/autodl-tmp/HDpainter1/model

/root/miniconda3/envs/czf/bin/python train.py \
  --run-name hdpainter_coad_prad_nsclc \
  --data-dir /root/autodl-tmp/OV/batch_preprocess/COAD/regularize_train_tiles_degraded_COAD_nmf48.instchunk512_train.maskrefined.h5 \
  --epochs 12 \
  --batch-size 2 \
  --num-workers 2 \
  --save-latest \
  --gpu-ids 0
```

Important training arguments:

- `--data-dir`: final `.maskrefined.h5` file or compatible dataset directory.
- `--epochs`: number of supervised training epochs.
- `--instance-batch-limit`: maximum number of instances per forward/backward micro-batch.
- `--canvas-size`, `--neighbor-k`, `--aggregate-radius`, `--boundary-samples`: model geometry and local context settings. Keep these consistent with inference unless intentionally changing the model family.

Checkpoints are written under:

```text
model/runs/<run-name>/checkpoints/
```

Do not commit the `model/runs/` directory.

## 4. Run Real Visium HD Inference

`inference/infer_hd.py` is the preferred one-command inference entry point. It can:

1. Check or generate nucleus labels from H&E.
2. Build the HDpainter inference H5.
3. Run the trained segmentation model.
4. Write predicted cell IDs back to the bin-level h5ad.
5. Aggregate bins into a cell-level h5ad.
6. Run `post_process.py`.
7. Optionally run `signal_process.py`.

Example:

```bash
cd /root/autodl-tmp/HDpainter1

/root/miniconda3/envs/czf/bin/python inference/infer_hd.py \
  --input-h5ad /root/autodl-tmp/OV/nucleus_segment/refer_OV_hd_nucleus_segmented.h5ad \
  --source-image-path /root/autodl-tmp/OV/Visium_HD_Human_Ovarian_Cancer_tissue_image.tif \
  --checkpoint /root/autodl-tmp/HDpainter1/model/runs/<run-name>/checkpoints/epoch_012.pt \
  --output-dir /root/autodl-tmp/OV/hdpainter_inference \
  --run-prefix refer_OV_hd \
  --basis-h5ad /root/autodl-tmp/OV/raw_sys_cell_nmf48.h5ad \
  --basis-varm-key NMF_H_48 \
  --nucleus-label-cols stardist_id,cellpose_id \
  --post-min-counts 20 \
  --post-min-genes 15 \
  --post-min-bins 9 \
  --signal-n-top-genes 0 \
  --signal-epochs 50 \
  --signal-num-batch-x 4 \
  --signal-num-batch-y 4
```

Use `--skip-signal-process` if only the segmentation result is needed.

Main outputs include:

```text
<output-dir>/<run-prefix>_<label-col>_pred.h5ad
<output-dir>/<run-prefix>_<label-col>_pred_cell_level.h5ad
<output-dir or signal output>/<run-prefix>_<label-col>_pred_cell_level_signal.h5ad
```

## 5. Post-Process and GNN Signal Enhancement

If a cell-level h5ad has already been generated, the post-processing stage can be run separately.

First inspect QC distributions:

```bash
/root/miniconda3/envs/czf/bin/python inference/post_data_check.py \
  --input-h5ad /root/autodl-tmp/OV/hdpainter_inference/refer_OV_hd_stardist_id_pred_cell_level.h5ad \
  --output-dir /root/autodl-tmp/HDpainter1/inference/data_check
```

Prepare model input:

```bash
/root/miniconda3/envs/czf/bin/python inference/post_process.py \
  --input-h5ad /root/autodl-tmp/OV/hdpainter_inference/refer_OV_hd_stardist_id_pred_cell_level.h5ad \
  --output-h5ad /root/autodl-tmp/OV/hdpainter_inference/refer_OV_hd_stardist_id_pred_cell_level_post.h5ad \
  --min-counts 20 \
  --min-genes 15 \
  --min-bins 9 \
  --signal-n-top-genes 0 \
  --skip-pca
```

Run graph signal processing:

```bash
/root/miniconda3/envs/czf/bin/python inference/signal_process.py \
  --input-h5ad /root/autodl-tmp/OV/hdpainter_inference/refer_OV_hd_stardist_id_pred_cell_level_post.h5ad \
  --output-h5ad /root/autodl-tmp/OV/gnn_signal/refer_OV_hd_stardist_id_signal_e50.h5ad \
  --dim-reduction HVG \
  --graph-input-dim 64 \
  --epochs 50 \
  --num-batch-x 4 \
  --num-batch-y 4 \
  --batch-spatial-k 4 \
  --batch-expression-k 3 \
  --key-added GNN
```

With `--signal-n-top-genes 0`, all genes are marked for the signal model. The graph model stores:

```text
obsm["GNN"]        # low-dimensional cell embedding
layers["GNN_ReX"]  # reconstructed full-gene expression
```

The current preferred GNN setting for around 100k cells is 4 x 4 DIC batching with 50 epochs for initial testing.

## 6. Validation and Comparison

The validation code is designed to compare HDpainter against bin2cell-style expansion and to evaluate signal quality after GNN enhancement.

Run bin2cell baseline validation on held-out synthetic HD tiles:

```bash
/root/miniconda3/envs/czf/bin/python validation/bin2cell_validation_baseline.py \
  --sample COAD \
  --sample PRAD \
  --sample NSCLC \
  --data-root /root/autodl-tmp/OV/batch_preprocess \
  --output-dir /root/autodl-tmp/HDpainter1/validation/bin2cell_baseline
```

Summarize validation results:

```bash
/root/miniconda3/envs/czf/bin/python validation/summarize_validation_suite.py \
  --project-root /root/autodl-tmp/HDpainter1 \
  --ov-root /root/autodl-tmp/OV \
  --validation-dir /root/autodl-tmp/HDpainter1/validation \
  --epoch 12
```

Compare bin2cell, direct HDpainter segmentation, and GNN-enhanced cell-level signals:

```bash
/root/miniconda3/envs/czf/bin/python inference/compare_data.py \
  --dataset bin2cell=/root/autodl-tmp/OV/hdpainter_inference/refer_OV_hd_stardist_id_pred_cell_level.h5ad:counts \
  --dataset direct=/root/autodl-tmp/OV/refer_OV_hd_pred_cell_level_epoch012.h5ad:counts \
  --dataset gnn_e50=/root/autodl-tmp/OV/gnn_signal_epoch012/refer_OV_hd_pred_cell_level_epoch012_signal_e50.h5ad:GNN_ReX \
  --reconstruction-dataset gnn_e50 \
  --output-dir /root/autodl-tmp/HDpainter1/validation/evaluation_suite/signal_compare
```

The current evaluation focus is:

- Held-out reconstruction quality.
- Marker-gene spatial quality.
- HDpainter validation IoU on Xenium-derived validation tiles.
- bin2cell default expansion IoU on the same validation tiles.
- Cell-level signal consistency after GNN reconstruction.

## Current Defaults

```text
NMF source: Xenium union-cell pseudo-single-cell matrix
NMF components: 48
Pseudo-cell QC: total_counts > 1, n_genes_by_counts > 3
Gene QC before NMF: expressing_cells > 3
HD degradation: foreground/background gamma-poisson from HD labels_he_expanded
Post-process QC: min_counts=20, min_genes=15, min_bins=9
Signal genes: all genes, signal_n_top_genes=0
Signal model: RGAT graph autoencoder
Signal DIC: 4 x 4 tiles
Signal test epochs: 50
Regularize workers: 8
Mask-refine workers: 8
Recommended memory limit for batch preprocessing: 62 GB
```

## Notes

- Do not run multiple samples concurrently during preprocessing unless disk, memory, and log separation have been explicitly checked.
- If `reference_hd_square_002um.h5ad` is missing, do not pass `--skip-reference`.
- If the HD reference lacks `labels_he_expanded`, preprocessing should derive it from `labels_he` or nucleus/cellpose labels; otherwise StarDist/bin2cell must be run with the H&E image.
- The current standard workflow must not fall back to full-bin NMF or HD-expanded pseudo-cell NMF.
- `signal_process.py` is intentionally separated from `post_process.py` so that graph model design can evolve without changing the basic cell-level AnnData preparation step.
