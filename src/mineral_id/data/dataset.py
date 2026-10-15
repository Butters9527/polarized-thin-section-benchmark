"""RockClassificationDataset：岩石薄片三大岩类图像分类 Dataset。"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class RockClassificationDataset(Dataset):
    """读取 split CSV，返回 PPL 单偏光图像与岩类标签。

    第一阶段只使用 photo_id=1（PPL 单偏光）的图像做三大岩类分类。
    标签映射：igneous→0 / metamorphic→1 / sedimentary→2。

    Args:
        csv_path: train.csv / val.csv / test.csv 的路径。
        mode: 'train' / 'val' / 'test'，仅作标记，不影响内部逻辑（transform 由调用方传入）。
        transform: torchvision transform，为 None 时只做 ToTensor。
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

        # 只保留 PPL 单偏光（photo_id == 1）
        self._df = raw_df[raw_df["photo_id"] == 1].reset_index(drop=True)

        self._transform = transform if transform is not None else transforms.ToTensor()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        row = self._df.iloc[idx]

        filepath = Path(row["filepath"])
        if not filepath.exists():
            raise FileNotFoundError(
                f"图像文件不存在（idx={idx}）：{filepath}\n"
                "请确认数据集路径是否挂载，或 CSV 中的 filepath 列是否正确。"
            )

        image = Image.open(filepath).convert("RGB")
        tensor = self._transform(image)
        label = self.CLASS_TO_IDX[row["rock_class"]]

        return tensor, label


# ----------------------------------------------------------------------
if __name__ == "__main__":
    from mineral_id.data.transforms import get_train_transform

    # 路径相对项目根目录运行
    csv_path = Path("data/processed/splits/train.csv")
    dataset = RockClassificationDataset(csv_path, transform=get_train_transform())

    print(f"样本总数（PPL only）：{len(dataset)}")
    tensor, label = dataset[0]
    print(f"第一个样本 —— shape: {tensor.shape}, label: {label} ({dataset.IDX_TO_CLASS[label]})")
