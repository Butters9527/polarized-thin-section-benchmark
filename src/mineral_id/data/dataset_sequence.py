"""SequenceDataset：PPL + 7 帧 XPL 消光序列 Dataset（photo_id 1–8）。"""

import warnings
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class SequenceDataset(Dataset):
    """读取 split CSV，按薄片组装 8 帧消光序列（photo_id 1–8），
    返回 (sequence_tensor [8,3,H,W], label) 用于时序建模。

    帧顺序：photo_id 1（PPL）、2–8（XPL @0°/15°/30°/45°/60°/75°/90°）。
    联合空间增强：8 帧共享同一随机种子，保证空间对齐。
    若薄片 photo_id 1–8 不完整，发出警告并跳过。

    标签映射：igneous→0 / metamorphic→1 / sedimentary→2。

    Args:
        csv_path: train.csv / val.csv / test.csv 的路径。
        mode: 'train' / 'val' / 'test'，仅作标记，不影响内部逻辑。
        transform: 作用于每帧的 transform；为 None 时只做 ToTensor。
    """

    CLASS_TO_IDX: dict[str, int] = {
        "igneous":     0,
        "metamorphic": 1,
        "sedimentary": 2,
    }
    IDX_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

    _REQUIRED_PHOTO_IDS: tuple[int, ...] = tuple(range(1, 9))  # 1–8

    def __init__(
        self,
        csv_path: str | Path,
        mode: str = "train",
        transform: transforms.Compose | None = None,
    ) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在：{csv_path}")

        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode 必须为 'train' / 'val' / 'test'，收到：{mode!r}")
        self.mode = mode

        self._transform = transform if transform is not None else transforms.ToTensor()

        df = pd.read_csv(csv_path)
        self._samples = self._build_sequences(df)

    # ------------------------------------------------------------------
    def _build_sequences(
        self, df: pd.DataFrame
    ) -> list[tuple[list[str], int]]:
        """构建 ([frame_path × 8], label) 样本列表。"""
        key_cols = ["rock_class", "rock_id", "slide_id"]
        groups = df.groupby(key_cols)

        samples: list[tuple[list[str], int]] = []
        for (rock_class, rock_id, slide_id), group in groups:
            present = set(group["photo_id"].tolist())
            missing = [pid for pid in self._REQUIRED_PHOTO_IDS if pid not in present]
            if missing:
                warnings.warn(
                    f"跳过薄片 [{rock_class} rock_id={rock_id} slide_id={slide_id}]："
                    f"缺少 photo_id={missing}",
                    stacklevel=2,
                )
                continue

            ordered = (
                group[group["photo_id"].isin(self._REQUIRED_PHOTO_IDS)]
                .sort_values("photo_id")
            )
            frame_paths = ordered["filepath"].tolist()
            label = self.CLASS_TO_IDX[rock_class]
            samples.append((frame_paths, label))

        return samples

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        frame_paths, label = self._samples[idx]

        seed = torch.randint(0, 2**31, (1,)).item()
        frames: list[Tensor] = []
        for path in frame_paths:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(
                    f"图像文件不存在（idx={idx}）：{p}\n"
                    "请确认数据集路径是否挂载，或 CSV 中的 filepath 列是否正确。"
                )
            img = Image.open(p).convert("RGB")
            torch.manual_seed(seed)
            frames.append(self._transform(img))  # [3, H, W]

        sequence_tensor = torch.stack(frames, dim=0)  # [8, 3, H, W]
        return sequence_tensor, label

    # ------------------------------------------------------------------


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from mineral_id.data.transforms import get_train_transform

    csv_path = Path("data/processed/splits/train.csv")
    dataset = SequenceDataset(csv_path, mode="train", transform=get_train_transform())

    print(f"样本总数（8 帧序列）：{len(dataset)}")
    seq, label = dataset[0]
    print(f"第一个样本 —— shape: {seq.shape}, label: {label} ({dataset.IDX_TO_CLASS[label]})")
