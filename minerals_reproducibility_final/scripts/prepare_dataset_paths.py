#!/usr/bin/env python3
"""Create local filepath CSVs from the public NJU dataset and the published split assignments.

Usage:
    python scripts/prepare_dataset_paths.py --dataset-root /path/to/NJU_dataset

The script recursively indexes image filenames below --dataset-root and writes local, machine-specific
CSV files under data/processed/. These generated files are intentionally git-ignored because they contain
local filesystem paths.
"""
from pathlib import Path
import argparse
import pandas as pd

IMAGE_EXTS={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}

def index_images(root: Path):
    idx={}
    dup=set()
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if p.name in idx: dup.add(p.name)
            else: idx[p.name]=p.resolve()
    if dup:
        raise RuntimeError(f'Duplicate filenames found; cannot resolve uniquely: {sorted(dup)[:10]}')
    return idx

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True, type=Path)
    args=ap.parse_args()
    root=args.dataset_root.resolve()
    index=index_images(root)

    repo=Path(__file__).resolve().parents[1]
    frame_meta=pd.read_csv(repo/'data/metadata/frame_metadata.csv')
    frame_meta['filepath']=frame_meta['filename'].map(lambda n: str(index.get(n,'')))
    missing=frame_meta.loc[frame_meta.filepath.eq(''),'filename'].tolist()
    if missing:
        raise FileNotFoundError(f'{len(missing)} published filenames were not found under {root}. First examples: {missing[:10]}')

    exp_dir=repo/'data/processed/dataset_exploration'; exp_dir.mkdir(parents=True,exist_ok=True)
    cols=['filepath']+[c for c in frame_meta.columns if c!='filepath']
    frame_meta[cols].to_csv(exp_dir/'dataset_summary.csv',index=False,encoding='utf-8-sig')
    pd.read_csv(repo/'data/metadata/slide_metadata.csv').to_csv(exp_dir/'slide_summary.csv',index=False,encoding='utf-8-sig')

    assignments=pd.read_csv(repo/'data/splits/fixed_frame_assignments.csv')
    assignments['filepath']=assignments['filename'].map(lambda n: str(index[n]))
    split_dir=repo/'data/processed/splits'; split_dir.mkdir(parents=True,exist_ok=True)
    outcols=['filepath','rock_class','rock_id','rock_name','slide_id','photo_id','polarization','rotation_angle']
    for split,df in assignments.groupby('split'):
        df[outcols].to_csv(split_dir/f'{split}.csv',index=False,encoding='utf-8-sig')
        print(f'{split}: {len(df)} frames -> {split_dir/f"{split}.csv"}')

if __name__=='__main__': main()
