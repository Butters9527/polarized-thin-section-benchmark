# Package Scope and Privacy Notes

This public package was curated specifically for the *Minerals* manuscript on NJU rock thin-section classification.

Included materials are limited to manuscript-related training/evaluation code, public split/fold assignments, lightweight numerical outputs, and figure-generation utilities. Raw image data and model weights are not redistributed.

Excluded materials include separate datasets/manuscripts, patent experiments, GUI/Web prototypes, internal review/audit files, temporary export utilities, local absolute filesystem paths, personal contact information, and unrelated development artifacts.

Run `scripts/prepare_dataset_paths.py` locally after downloading the public NJU dataset to create machine-specific path files under `data/processed/`.

The software versions listed in `README.md` are a directly verified snapshot of the current project virtual environment on 2026-09-29. The backbone-comparison batch sizes were additionally checked against the actual PowerShell command history: ResNet-50 = 16, EfficientNet-B0 = 16, ViT-B/16 = 8, and Swin-T = 16.
