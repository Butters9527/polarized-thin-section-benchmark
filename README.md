# Reproducibility Materials for Polarized-Light Rock Thin-Section Classification

This repository provides the code, data-split assignments, experiment summaries, and figure-generation utilities associated with the manuscript:

**“Evaluating Polarized-Light Input Representations for Rock Thin Section Classification: A Benchmark of Dual-Modal Fusion and Extinction-Sequence Modeling”**

The repository covers the NJU rock thin-section experiments reported in the manuscript, including the single-frame PPL baseline, PPL+XPL early fusion, multi-frame extinction-sequence models, cross-validation analyses, hyperparameter sensitivity analysis, backbone comparison, and interpretability experiments.

Raw thin-section images and trained model weights are not redistributed. The source images are available from the public **NJU Rock Thin Section Microscopic Image Dataset** hosted by Science Data Bank:

**DOI:** 10.11922/sciencedb.j00001.00097

---

## Repository Structure

```text
data/
├── metadata/                 Frame- and slide-level metadata
└── splits/                   Fixed and cross-validation assignments

src/mineral_id/
├── data/                     Dataset loading and preprocessing
├── models/                   Classification and sequence models
└── training/                 Main training entry points

scripts/
├── Evaluation utilities
├── Cross-validation scripts
├── Hyperparameter search
├── Backbone comparison
├── Grad-CAM visualization
├── Attention-weight analysis
└── Dataset-path preparation

experiments/
└── Metrics, logs, and lightweight numerical outputs used in the manuscript

figure_scripts_NJU/
└── Scripts and numerical inputs used to generate manuscript figures
```

Machine-specific filesystem paths have been removed from the released metadata and split files.

---

## 1. Prepare the NJU Dataset

Download and extract the public NJU dataset, then run:

```bash
python scripts/prepare_dataset_paths.py --dataset-root /path/to/NJU_dataset
```

The script reconstructs local image paths and writes the resulting CSV files under:

```text
data/processed/
```

These files are not included in the repository because they contain machine-specific absolute paths.

The manuscript uses a slide-level stratified fixed split after excluding three incomplete image sequences. The released assignments are:

```text
data/splits/fixed_slide_assignments.csv
data/splits/fixed_frame_assignments.csv
```

Cross-validation assignments are provided in:

```text
data/splits/kfold_assignments.csv
data/splits/nested_cv_assignments.csv
```

The split files allow the sample membership of each training, validation, test, and cross-validation partition to be inspected independently of the local image paths.

---

## 2. Software Environment

The released code was verified in the following project environment on 29 September 2026.

| Component | Version |
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
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM |

These versions represent the environment used to verify the released code. Minor third-party library versions may differ from those used in some of the original training runs because a historical environment lock file was not retained.

Install PyTorch and torchvision for the appropriate CUDA platform first, and then install the project and remaining dependencies:

```bash
pip install -e .
```

The experiments reported in the manuscript were conducted on an NVIDIA CUDA-enabled GPU. The complete training workflow has not been systematically tested in a CPU-only environment.

---

## 3. Main Experiments

Run all commands from the repository root after preparing `data/processed/`.

### exp01 — Single-frame PPL baseline

```bash
python -m mineral_id.training.train --batch-size 16 --seed 42
```

### exp02 — PPL + XPL@0° early fusion

```bash
python -m mineral_id.training.train_dual --batch-size 16 --seed 42
```

### exp03 — Eight-frame sequence model

The sequence consists of one PPL frame and seven XPL frames acquired from 0° to 90° at 15° intervals.

```bash
python -m mineral_id.training.train_sequence --batch-size 8
```

### exp03b — Regularized sequence model

```bash
python -m mineral_id.training.train_sequence_b --batch-size 8 --dropout 0.5
```

### exp04 — Sequence model with rotational positional encoding

```bash
python -m mineral_id.training.train_sequence_rpe
```

The `experiments/` directory contains the lightweight metrics and logs used in the manuscript. Model checkpoints are not included.

---

## 4. Cross-Validation

### Non-nested five-fold cross-validation

```bash
python scripts/kfold_cv.py --mode exp01
python scripts/kfold_cv.py --mode exp02
```

### Nested five-fold cross-validation

```bash
python scripts/nested_cv.py --mode exp01
python scripts/nested_cv.py --mode exp02
```

The nested analysis separates the inner validation split used for model selection and early stopping from the outer evaluation fold.

Fold membership is also provided explicitly in:

```text
data/splits/kfold_assignments.csv
data/splits/nested_cv_assignments.csv
```

---

## 5. Hyperparameter Sensitivity Analysis

Run:

```bash
python scripts/hyperparam_search.py
```

The corresponding lightweight outputs are included in the repository and support the hyperparameter sensitivity analysis reported in the manuscript.

---

## 6. Backbone Comparison

The backbone comparison uses the same dual-modal PPL+XPL@0° early-fusion framework while replacing the ImageNet-pretrained feature extractor.

The batch sizes used in the reported experiments were:

| Backbone | Batch size |
|---|---:|
| ResNet-50 | 16 |
| EfficientNet-B0 | 16 |
| ViT-B/16 | 8 |
| Swin-T | 16 |

Run the experiments with explicit batch-size arguments:

```bash
python scripts/backbone_comparison.py --backbone resnet50 --batch-size 16
python scripts/backbone_comparison.py --backbone efficientnet_b0 --batch-size 16
python scripts/backbone_comparison.py --backbone vit_b_16 --batch-size 8
python scripts/backbone_comparison.py --backbone swin_t --batch-size 16
```

The same data split, preprocessing, augmentation procedure, optimizer, learning-rate schedule, weight decay, and early-stopping protocol were used across the four backbones. ViT-B/16 used a batch size of 8 because of its higher GPU-memory requirement on the 8 GB GPU; the other three backbones used a batch size of 16.

The resulting summary is provided in:

```text
experiments/backbone_comparison_summary.csv
```

---

## 7. Evaluation and Interpretability

The repository includes utilities for held-out evaluation, Grad-CAM visualization, and frame-level attention analysis.

Examples:

```bash
python scripts/evaluate.py --help
python scripts/gradcam_viz.py --help
python scripts/attention_viz.py --help
```

Checkpoint-dependent analyses require a locally trained model checkpoint.

Grad-CAM and attention-weight outputs used in the manuscript are represented by the lightweight numerical or image-derived materials included under `experiments/` and the corresponding figure-generation scripts.

---

## 8. Figure Generation

Scripts used to produce the manuscript figures are located in:

```text
figure_scripts_NJU/
```

The figures were generated from the experimental outputs and source images using Python-based plotting and image-processing scripts.

Where a figure depends on a trained model, the relevant checkpoint must first be reproduced locally.

---

## 9. Reproducibility Notes

- The released split files preserve the sample assignments used in the manuscript.
- Raw NJU images must be downloaded separately from Science Data Bank.
- Model checkpoints are not included because of their size.
- Numerical metrics and lightweight logs supporting the manuscript tables and analyses are included where practical.
- Re-logged training curves are provided to document training dynamics; manuscript performance values are based on the retained checkpoints from the corresponding experiments.
- Exact bitwise reproducibility is not expected across different GPUs, CUDA kernels, drivers, or library builds.
- Small numerical differences between reruns may therefore occur even when the same random seed and data partition are used.

---

## 10. Dataset Citation

If you use the NJU thin-section images, please cite the original dataset using the citation information provided by Science Data Bank.

Dataset DOI:

```text
10.11922/sciencedb.j00001.00097
```

---

## 11. Manuscript

These materials accompany:

> **Evaluating Polarized-Light Input Representations for Rock Thin Section Classification: A Benchmark of Dual-Modal Fusion and Extinction-Sequence Modeling**

The repository is intended to support inspection and reproduction of the experimental procedures and results reported in that study.
