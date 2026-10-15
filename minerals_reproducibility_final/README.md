# Reproducibility Materials: Polarized-Light Rock Thin-Section Classification

This repository contains the code, split assignments, lightweight experiment outputs, and figure-generation utilities supporting the manuscript **“Evaluating Polarized-Light Input Representations for Rock Thin Section Classification: A Benchmark of Dual-Modal Fusion and Extinction-Sequence Modeling”**, submitted to *Minerals*.

## Scope

This package is intentionally limited to the NJU rock thin-section experiments reported in this manuscript. It does **not** contain materials from other datasets, manuscripts, patent experiments, or software prototypes.

Raw thin-section images and trained model weights are not redistributed. The source images are available from the public **NJU Rock Thin Section Microscopic Image Dataset** at Science Data Bank (DOI: `10.11922/sciencedb.j00001.00097`).

## Repository Contents

- `data/metadata/`: frame- and slide-level metadata with machine-local filesystem paths removed.
- `data/splits/`: fixed train/validation/test assignments, non-nested 5-fold assignments, and nested-CV assignments used for the manuscript analyses.
- `src/mineral_id/`: dataset, model, and training modules for the PPL baseline, PPL+XPL@0° early fusion, and 8-frame sequence models.
- `scripts/`: evaluation, cross-validation, hyperparameter-search, backbone-comparison, attribution, attention, and dataset-path preparation scripts.
- `experiments/`: lightweight metrics and logs supporting the reported tables and analyses. Model checkpoints are excluded.
- `figure_scripts_NJU/`: scripts and small numeric inputs used to reproduce manuscript figures.

## 1. Prepare the Public NJU Dataset

After downloading and extracting the NJU dataset, create machine-local CSV files containing the real image paths:

```bash
python scripts/prepare_dataset_paths.py --dataset-root /path/to/NJU_dataset
```

The generated files are written under `data/processed/`. They are intentionally excluded from version control because they contain machine-specific absolute paths.

The manuscript uses a slide-level stratified fixed split after exclusion of three incomplete sequences. The public assignments are stored in:

- `data/splits/fixed_slide_assignments.csv`
- `data/splits/fixed_frame_assignments.csv`

The non-nested and nested 5-fold assignments are stored in:

- `data/splits/kfold_assignments.csv`
- `data/splits/nested_cv_assignments.csv`

## 2. Verified Software Environment

The following environment was directly verified in the project virtual environment on **2026-09-29** and is the recommended reference stack for running the released code:

| Component | Verified version |
|---|---|
| Python | 3.12.9 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| CUDA runtime reported by PyTorch | 12.1 |
| NumPy | 2.4.4 |
| pandas | 3.0.3 |
| Pillow | 12.2.0 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.9 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) |

This table is a verified snapshot of the current project environment. It should not be interpreted as proof that every third-party library had exactly the same patch version on the date of every original training run unless an archived environment lock file is available.

Install PyTorch/torchvision for an appropriate CUDA platform, then install the package and remaining dependencies:

```bash
pip install -e .
```

The experiments reported in the manuscript were run on an NVIDIA CUDA-enabled GPU. CPU-only execution of the complete training workflow has not been systematically validated.

## 3. Main Training Entry Points

Run the following commands from the repository root after preparing `data/processed/`.

### exp01 — single-frame PPL baseline

```bash
python -m mineral_id.training.train --batch-size 16 --seed 42
```

### exp02 — PPL + XPL@0° early fusion

```bash
python -m mineral_id.training.train_dual --batch-size 16 --seed 42
```

### exp03 — 8-frame sequence model

```bash
python -m mineral_id.training.train_sequence --batch-size 8
```

### exp03b — regularized/frozen sequence model

```bash
python -m mineral_id.training.train_sequence_b --batch-size 8 --dropout 0.5
```

### exp04 — sequence model with rotational positional encoding

```bash
python -m mineral_id.training.train_sequence_rpe
```

The archived lightweight outputs under `experiments/` correspond to the manuscript analyses. Model checkpoints are not included in this public package.

## 4. Cross-Validation and Sensitivity Analyses

### Non-nested 5-fold cross-validation

```bash
python scripts/kfold_cv.py --mode exp01
python scripts/kfold_cv.py --mode exp02
```

### Nested 5-fold cross-validation

```bash
python scripts/nested_cv.py --mode exp01
python scripts/nested_cv.py --mode exp02
```

### Hyperparameter sensitivity analysis

```bash
python scripts/hyperparam_search.py
```

The released assignment tables make the fold membership inspectable independently of local filesystem paths.

## 5. Backbone Comparison

The backbone-comparison script uses the same dual-modal PPL+XPL@0° early-fusion setup while replacing the ImageNet-pretrained backbone. The actual commands used were checked against the local PowerShell command history. The resulting batch sizes were:

| Backbone | Batch size |
|---|---:|
| ResNet-50 | 16 |
| EfficientNet-B0 | 16 |
| ViT-B/16 | 8 |
| Swin-T | 16 |

Use explicit batch-size arguments to avoid relying on defaults:

```bash
python scripts/backbone_comparison.py --backbone resnet50 --batch-size 16
python scripts/backbone_comparison.py --backbone efficientnet_b0 --batch-size 16
python scripts/backbone_comparison.py --backbone vit_b_16 --batch-size 8
python scripts/backbone_comparison.py --backbone swin_t --batch-size 16
```

ViT-B/16 used a smaller batch size because of its higher GPU-memory requirement on the 8 GB GPU. Therefore, the backbone comparison should be described as using matched data, preprocessing, augmentation, optimizer/schedule, weight decay, and early-stopping settings **except for batch size**.

The corresponding lightweight summary is provided in `experiments/backbone_comparison_summary.csv`.

## 6. Evaluation and Visualization Utilities

The package also includes utilities for held-out evaluation, Grad-CAM visualization, attention-weight analysis, and manuscript figure generation. These scripts require the relevant locally trained checkpoint where applicable.

Examples:

```bash
python scripts/evaluate.py --help
python scripts/gradcam_viz.py --help
python scripts/attention_viz.py --help
```

Figure-generation scripts are located in `figure_scripts_NJU/`.

## 7. Reproducibility Notes

- The package contains the public split/fold assignments and lightweight numerical outputs used to support the manuscript tables and plots.
- Raw NJU images must be downloaded separately from the public dataset source.
- Model checkpoints are excluded because of file size; checkpoint-dependent analyses require locally trained weights.
- Exact GPU-level bitwise reproducibility is not guaranteed across different CUDA kernels, drivers, GPUs, or library builds.
- Re-logged training curves in the package are for documenting training dynamics and should not be interpreted as replacing the retained checkpoint metrics used in the manuscript.
- Machine-local absolute paths, personal contact information, unrelated project materials, and internal workflow files have been removed from this public package.

## 8. Privacy and Package Boundary

This release was curated specifically for the *Minerals* manuscript. It excludes materials from separate datasets/manuscripts, patent-related experiments, GUI/Web prototypes, internal review/audit files, local absolute paths, personal contact details, and unrelated development artifacts.
