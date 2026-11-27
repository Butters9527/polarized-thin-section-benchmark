# R2 materials (manuscript minerals-4666317, revised version)

This folder contains everything needed to inspect and reproduce the results of the revised manuscript
“Polarized-Light Rock Thin-Section Classification under Rock-Type-Disjoint Cross-Validation: Effects of Input
Representation and Backbone Architecture”. All 1280 models were trained with the settings below, and every number,
table and figure in the manuscript is computed from the per-section predictions released here.

## Contents

| Path | Content |
|---|---|
| `code/build_index.py` | Builds the section index of the 321 usable NJU sections and the resized image caches |
| `code/train_r2.py` | Trains and evaluates all models (one JSON file per run) |
| `code/analyze_r2.py` | Summary statistics: fold OA, macro-F1, per-class recall, corrected resampled t-test, cluster bootstrap, McNemar tests |
| `code/locality_groups.py` | Recomputes the rock-name + locality groups from `data/section_locality.csv` and checks the locality folds |
| `code/run_all.sh` | Runs the complete pipeline |
| `code/requirements.txt` | Python packages |
| `data/sections.csv` | The 321 usable sections: rock class, rock-type number, rock name, grouping key, image size, resolution group, relative paths of the eight frames, original fixed-split role |
| `data/folds.csv` | All partitions: `protocol` × `rep` (0–9) × `fold` (0–4) × section → `role` (`train`, `inner_val` = early-stopping set, `test`) |
| `data/section_locality.csv` | Locality of each section (from the information tables distributed with the NJU dataset) and its rock-name + locality group |
| `data/fixed_frame_assignments.csv` | The fixed 256/32/33 split |
| `results/r2_runs.zip` | Per-run JSON files of all 1280 runs (settings, seed, training history per epoch, selected epoch, per-section test predictions and probabilities, five-crop and channel-occlusion predictions, positional-encoding statistics) |
| `analysis/cv_summary.csv` | Cross-validation summaries (manuscript Tables 1, 2 and 4) |
| `analysis/pairwise.csv` | Pairwise comparisons (Table 3) |
| `analysis/fixed_summary.csv` | Fixed split, ten seeds (Table 5) |
| `analysis/curves_fixed.csv` | Training curves on the fixed split (Figure 6) |
| `analysis/oof_predictions.csv` | Out-of-fold predictions of every cross-validation run |
| `figures/` | Scripts for Figures 1–6 |

Raw NJU images and model checkpoints are not redistributed. The images are available from Science Data Bank,
doi:10.11922/sciencedb.j00001.00097.

## Protocols (`protocol` in folds.csv; `setting` in results and analysis files)

| Code | Protocol |
|---|---|
| `grouped` / `grouped_orig` | Five-fold cross-validation grouped by rock name (103 groups), repeated ten times; primary protocol |
| `grouped_norm` | Same folds, resolution-normalized inputs |
| `locality` / `locality_orig` | Five-fold cross-validation grouped by rock name and locality (77 connected groups), repeated ten times |
| `slide` / `slide_orig` | Section-level stratified five-fold cross-validation, repeated ten times |
| `fixed_orig` | Fixed 256/32/33 split, ten seeds |

In every cross-validation fold, about 15% of the training sections (whole groups in the grouped protocols) form an
early-stopping set (`inner_val`) that is never used for evaluation. No hyperparameters are tuned in the folds.

## Models (`model` codes)

| Code | Manuscript name |
|---|---|
| `ppl`, `xpl0`, `dual` | PPL, XPL (XPL@0°), PPL+XPL (six-channel early fusion), ResNet-50 |
| `dual_mis` | PPL+XPL-Mis (independent augmentation of the two frames; fixed split only) |
| `seq_mean`, `seq_attn` | Seq-Mean, Seq-Attn (eight frames, shared ResNet-50) |
| `seq_rpe90` | Seq-Attn-RPE, 90°-periodic rotational positional encoding, learnable gain initialized at 0.1 |
| `seq_rpe90_g03`, `seq_rpe90_g1` | Seq-Attn-RPE with the gain fixed at 0.3 and 1.0 |
| `xpl7_attn` | XPL7-Attn (seven XPL frames) |
| `effb0_ppl`, `effb0_dual`, `vitb16_ppl`, `vitb16_dual`, `swint_ppl`, `swint_dual` | PPL and PPL+XPL with EfficientNet-B0, ViT-B/16 and Swin-T |

The rotational positional encoding is implemented in `code/train_r2.py`: for an XPL frame at angle θ, the k-th
feature-dimension pair receives g·sin(2π m_k θ/90°) and g·cos(2π m_k θ/90°) with m_k = 1 + (k mod 3), so that 0° and
90° receive the same code. At 15° sampling the sine component of the third harmonic is zero at every sampled angle.

## Training settings (all models)

ImageNet-1K-pretrained backbones, all layers fine-tuned; AdamW (learning rate 1e-4, weight decay 1e-4); cosine decay
over at most 30 epochs; batch size 16; dropout 0.3; bfloat16 mixed precision; early stopping with patience 5 on the
early-stopping accuracy (ties broken by the lower loss). Images: shorter side resized to 256 pixels with the aspect
ratio preserved, random 224 × 224 crop, random horizontal and vertical flips and random rotation of up to ±90°
(identical parameters for all frames of a sample, except in `dual_mis`); evaluation on the center crop and, in
addition, on five crops. Runs were performed with PyTorch 2.5.1 on one NVIDIA RTX 4090 (24 GB). The seed of each run
is stored in its JSON file (`job.seed`). Bitwise identical results are not expected on other hardware.

## Reproduction

```bash
cd r2/code
pip install -r requirements.txt
bash run_all.sh /path/to/NJU_dataset          # about 5 h of training time on one RTX 4090 for all 1280 runs
```

To check the released statistics without retraining:

```bash
cd r2
python -m zipfile -e results/r2_runs.zip .    # creates results/<setting>/<model>/*.json
python code/analyze_r2.py --results results --index data --out analysis_check
python code/locality_groups.py --index data
```

Figures 4–6 are drawn from the released runs; Figures 2–3 need no data; Figure 1 needs the NJU images:

```bash
python figures/make_figs.py --index data --results results --out figures_out
python figures/make_schematics.py --out figures_out
python figures/make_fig1.py --nju /path/to/NJU_dataset --out figures_out
```

The figure scripts require Times New Roman and stop if it is not installed.

## Statistics

Cross-validation metrics are reported as mean ± sample standard deviation over the 50 test folds. Models evaluated
on the same folds are compared with the corrected resampled t-test (Nadeau and Bengio 2003; df = 49), supplemented by
a paired cluster bootstrap over rock-name groups (locality groups for the locality protocol; 2000 resamples), which is
conditional on the trained models. On the fixed split, models are compared per seed with the exact McNemar test.
