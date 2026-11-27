# Reproducibility Materials for Polarized-Light Rock Thin-Section Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23388853.svg)](https://doi.org/10.5281/zenodo.23388853)

This repository accompanies the manuscript

**“Polarized-Light Rock Thin-Section Classification under Rock-Type-Disjoint Cross-Validation: Effects of Input
Representation and Backbone Architecture”** (*Minerals*, manuscript minerals-4666317, revised version).

## Current version (v2.0.0): `r2/`

All results of the revised manuscript are in **[`r2/`](r2/README.md)**:

- fold assignments of all evaluation protocols: rock-name-grouped five-fold cross-validation repeated ten times
  (primary), rock-name + locality grouped cross-validation, section-level random cross-validation, and the fixed
  256/32/33 split trained with ten seeds;
- per-section predictions, training histories and settings of all 1280 trained models;
- the training, analysis and figure scripts, with one set of training settings for all models;
- the corrected rotational positional encoding with a 90° period;
- the summary tables used in the manuscript, which can be recomputed from the released predictions.

See [`r2/README.md`](r2/README.md) for the file descriptions, code tables and reproduction commands.

## Materials of the first submission (v1.0.x, superseded)

The top-level folders `src/`, `scripts/`, `experiments/`, `figure_scripts_NJU/` and `data/` document the first
submitted version of the manuscript (titled “Evaluating Polarized-Light Input Representations for Rock Thin Section
Classification: A Benchmark of Dual-Modal Fusion and Extinction-Sequence Modeling”). They are kept unchanged for
traceability but are superseded by `r2/`. The following issues of the first version were corrected in the revision:

- The rotational positional encoding in `src/mineral_id/training/train_sequence_rpe.py` uses sin(πθ/90°) and
  cos(πθ/90°), i.e. a 180° period, so 0° and 90° receive opposite codes. `r2/code/train_r2.py` uses a 90° period
  with first- to third-order harmonics.
- The evaluation used section-level splits, in which sister sections of the same rock type can fall on both sides of
  the split. The revision uses rock-name-grouped cross-validation as the primary protocol.
- The protocol described as “nested” cross-validation used its inner split only for early stopping; the revision
  describes it as cross-validation with an inner early-stopping split.
- Models were trained with different settings (batch size, frozen layers, dropout, learning rate); the revision uses
  one set of settings fixed in advance for all models.
- The training curves in `figure_scripts_NJU/` come from separately re-logged runs; in the revision, curves and
  results come from the same runs.
- The hyperparameter search and the Grad-CAM and attention-weight analyses are not part of the revised manuscript.

## Data

Raw thin-section images and trained model weights are not redistributed. The images are available from the public
NJU Rock Thin Section Microscopic Image Dataset, Science Data Bank, doi:10.11922/sciencedb.j00001.00097. Please cite
the dataset as indicated by Science Data Bank.

## License

The code in this repository is released under the MIT License; see `LICENSE`.
