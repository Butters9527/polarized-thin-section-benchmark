# Package Scope and Privacy Notes

This public package was curated specifically for the *Minerals* manuscript on NJU rock thin-section classification.

Included materials are limited to manuscript-related training/evaluation code, public split/fold assignments, lightweight numerical outputs, and figure-generation utilities. Raw image data and model weights are not redistributed.

Excluded materials include separate datasets/manuscripts, patent experiments, GUI/Web prototypes, internal review/audit files, temporary export utilities, local absolute filesystem paths, personal contact information, and unrelated development artifacts.

Run `scripts/prepare_dataset_paths.py` locally after downloading the public NJU dataset to create machine-specific path files under `data/processed/`.

The software versions listed in `README.md` are a directly verified snapshot of the current project virtual environment on 2026-09-29. The backbone-comparison batch sizes were additionally checked against the actual PowerShell command history: ResNet-50 = 16, EfficientNet-B0 = 16, ViT-B/16 = 8, and Swin-T = 16.

## Changes in v1.0.1 (2026-10-15)

- **exp03 re-logged training history corrected.** The history shipped in v1.0.0
  (`experiments/exp03_sequence_relog/` and its copy under `figure_scripts_NJU/`) came from a re-run that did not
  pass `dropout` explicitly and therefore used the model's class default of 0.5, whereas the original exp03
  checkpoint reported in the manuscript used dropout 0.3. It has been replaced by
  `experiments/exp03_sequence_relog_d03/` (dropout 0.3; 14 epochs, best validation accuracy 90.62% at epoch 9),
  which is the history shown in manuscript Figure 7. No reported test-set number is affected.
- `scripts/train_sequence_relog.py` now passes `--dropout 0.3` explicitly and writes to `exp03_sequence_relog_d03`.
- `figure_scripts_NJU/generate_training_curves.py` uses the corrected history by default and checks every
  training log against the epoch counts and best validation accuracies stated in the manuscript figure captions.
- Added the `LICENSE` file (MIT) referenced by `pyproject.toml`, which was missing from v1.0.0.
- `MANIFEST.csv` regenerated.

## Changes in v1.0.2 (2026-10-15)

- `src/mineral_id/training/train_sequence.py` (exp03) now passes `--dropout` explicitly, with a default of 0.3,
  the value used for the exp03 checkpoint reported in the manuscript. Previously it relied on the
  `TemporalAttentionModel` class default, which is 0.5 (the exp03b/exp04 value), so re-running exp03 from this
  repository would have used the wrong dropout. The README command for exp03 now states `--dropout 0.3`.
- `MANIFEST.csv` regenerated.
