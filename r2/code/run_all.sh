#!/bin/bash
# Reproduce all R2 runs and statistics.
# usage (from r2/code):  bash run_all.sh /path/to/NJU_dataset   [extra train_r2.py options, e.g. --stage A]
set -e
cd "$(dirname "$0")"
NJU=${1:?usage: bash run_all.sh /path/to/NJU_dataset}
shift
# 1. image caches and section index (written to ../work, not to ../data)
python build_index.py --data "$NJU" --fixed ../data/fixed_frame_assignments.csv --out ../work
# 2. use the released fold assignments (they include the rock-name + locality protocol)
cp ../data/folds.csv ../data/section_locality.csv ../work/
# 3. training and evaluation (one JSON per run under ../results/<setting>/<model>/)
python train_r2.py --index ../work --results ../results --stage all "$@"
# 4. statistics (tables in ../analysis)
python analyze_r2.py --results ../results --index ../work --out ../analysis
