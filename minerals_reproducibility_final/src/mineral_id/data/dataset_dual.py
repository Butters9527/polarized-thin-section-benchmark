"""DualStreamDataset：PPL + XPL 双流早期融合 Dataset。"""

import warnings
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class DualStreamDataset(Dataset):
    """读取 split CSV，配对 PPL (photo_id=1) 与 XPL@0° (photo_id=2)，
    拼接为 6 通道张量用于双流早期融合分类。

    唯一键：(rock_class, rock_id, slide_id) 三列组合，避免不同岩类相同编号错误配对。
    若某薄片缺少 photo_id=1 或 photo_id=2，跳过并发出警告。

    返回：(dual_tensor [6,H,W], label)
    标签映射：igneous→0 / metamorphic→1 / sedimentary→2。

    Args:
        csv_path: train.csv / val.csv / test.csv 的路径。
        mode: 'train' / 'val' / 'test'，仅作标记，不影响内部逻辑。
        transform: 分别作用于 PPL 和 XPL 两张图的 transform；为 None 时只做 ToTensor。
    """

    CLASS_TO_IDX: dict[str, int] = {
        "igneous":     0,
        "metamorphic": 1,
        "sedimentary": 2,
    }
    IDX_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

    def __init__(
        self,
        csv_path: str | Path | None = None,
        mode: str = "train",
        transform: transforms.Compose | None = None,
        *,
        df: pd.DataFrame | None = None,
        align_augmentation: bool = True,
    ) -> None:
        if df is not None:
            raw_df = df
        elif csv_path is not None:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV 文件不存在：{csv_path}")
            raw_df = pd.read_csv(csv_path)
        else:
            raise ValueError("csv_path 和 df 不能同时为 None")

        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode 必须为 'train' / 'val' / 'test'，收到：{mode!r}")
        self.mode = mode

        self._transform = transform if transform is not None else transforms.ToTensor()
        self._align_augmentation = align_augmentation
        self._samples = self._build_pairs(raw_df)

    # ------------------------------------------------------------------
    def _build_pairs(self, df: pd.DataFrame) -> list[tuple[str, str, int]]:
        """构建 (ppl_path, xpl_path, label) 样本列表。"""
        key_cols = ["rock_class", "rock_id", "slide_id"]
        ppl = df[df["photo_id"] == 1].set_index(key_cols)
        xpl = df[df["photo_id"] == 2].set_index(key_cols)

        samples: list[tuple[str, str, int]] = []
        all_keys = set(ppl.index) | set(xpl.index)

        for key in sorted(all_keys):
            rock_class, rock_id, slide_id = key
            has_ppl = key in ppl.index
            has_xpl = key in xpl.index

            if not has_ppl or not has_xpl:
                missing = "photo_id=1(PPL)" if not has_ppl else "photo_id=2(XPL)"
                warnings.warn(
                    f"跳过薄片 [{rock_class} rock_id={rock_id} slide_id={slide_id}]：缺少 {missing}",
                    stacklevel=2,
                )
                continue

            ppl_path = ppl.loc[key, "filepath"]
            xpl_path = xpl.loc[key, "filepath"]
            label = self.CLASS_TO_IDX[rock_class]
            samples.append((str(ppl_path), str(xpl_path), label))

        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        ppl_path, xpl_path, label = self._samples[idx]

        ppl_img = self._load_image(ppl_path, idx, "PPL")
        xpl_img = self._load_image(xpl_path, idx, "XPL")

        if self._align_augmentation:
            seed = torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(seed)
            ppl_tensor = self._transform(ppl_img)   # [3, H, W]
            torch.manual_seed(seed)
            xpl_tensor = self._transform(xpl_img)   # [3, H, W]
        else:
            ppl_tensor = self._transform(ppl_img)   # [3, H, W]
            xpl_tensor = self._transform(xpl_img)   # [3, H, W]

        dual_tensor = torch.cat([ppl_tensor, xpl_tensor], dim=0)  # [6, H, W]
        return dual_tensor, label

    # ------------------------------------------------------------------
    def _load_image(self, path: str, idx: int, stream: str) -> Image.Image:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"图像文件不存在（idx={idx}, {stream}）：{p}\n"
                "请确认数据集路径是否挂载，或 CSV 中的 filepath 列是否正确。"
            )
        return Image.open(p).convert("RGB")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from mineral_id.data.transforms import get_val_transform

    csv_path = Path("data/processed/splits/train.csv")
    dataset = DualStreamDataset(csv_path, mode="train", transform=get_val_transform())

    print(f"样本总数（PPL+XPL 配对）：{len(dataset)}")
    tensor, label = dataset[0]
    print(f"第一个样本 —— shape: {tensor.shape}, label: {label} ({dataset.IDX_TO_CLASS[label]})")
