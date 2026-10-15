#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
岩石薄片数据集 train/val/test 划分脚本
- 基于薄片唯一键 (rock_class, rock_id, slide_id) 进行分层抽样
- 划分比例 train:val:test = 80:10:10
- 排除指定的异常薄片
- 输出三个 CSV 文件，包含每个图像的基本信息
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# ==================== 配置 ====================
# 输入文件路径（相对于项目根目录）
SLIDE_SUMMARY_PATH = Path("data/processed/dataset_exploration/slide_summary.csv")
DATASET_SUMMARY_PATH = Path("data/processed/dataset_exploration/dataset_summary.csv")

# 输出目录
OUTPUT_DIR = Path("data/processed/splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 随机种子
RANDOM_SEED = 42

# 需要排除的异常薄片列表 (rock_class, rock_id, slide_id)
# 注意：rock_class 应与 slide_summary.csv 中的值一致（如 'sedimentary'）
EXCLUDED_SLIDES = [
    ('sedimentary', 12, 3),   # 钙质粉砂岩
    ('sedimentary', 16, 1),   # 页岩
    ('sedimentary', 9,  1),   # 长石砂岩
]

# 划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 需要输出的列（顺序固定）
OUTPUT_COLUMNS = [
    "filepath", "rock_class", "rock_id", "rock_name",
    "slide_id", "photo_id", "polarization", "rotation_angle"
]


def load_slide_summary():
    """读取薄片摘要 CSV，并添加唯一键列"""
    if not SLIDE_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"未找到薄片摘要文件: {SLIDE_SUMMARY_PATH}")
    df = pd.read_csv(SLIDE_SUMMARY_PATH)
    # 确保必要的列存在
    required_cols = ["rock_class", "rock_id", "slide_id", "rock_name"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"slide_summary.csv 缺少必要列: {col}")
    # 添加唯一键列（用于匹配）
    df["slide_key"] = df["rock_class"] + "_" + df["rock_id"].astype(str) + "_" + df["slide_id"].astype(str)
    return df


def load_dataset_summary():
    """读取图像摘要 CSV，并添加与薄片相同的唯一键列"""
    if not DATASET_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"未找到图像摘要文件: {DATASET_SUMMARY_PATH}")
    df = pd.read_csv(DATASET_SUMMARY_PATH)
    # 检查必要列
    required_cols = ["filepath", "rock_class", "rock_id", "slide_id", "photo_id",
                     "polarization", "rotation_angle", "rock_name"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"dataset_summary.csv 缺少必要列: {col}")
    # 添加唯一键列
    df["slide_key"] = df["rock_class"] + "_" + df["rock_id"].astype(str) + "_" + df["slide_id"].astype(str)
    return df


def exclude_abnormal_slides(df_slides, excluded_list):
    """从薄片 DataFrame 中排除指定的异常薄片"""
    initial_count = len(df_slides)
    for rock_class, rock_id, slide_id in excluded_list:
        mask = (df_slides["rock_class"] == rock_class) & \
               (df_slides["rock_id"] == rock_id) & \
               (df_slides["slide_id"] == slide_id)
        if mask.any():
            df_slides = df_slides[~mask]
            print(f"已排除异常薄片: rock_class={rock_class}, rock_id={rock_id}, slide_id={slide_id}")
        else:
            print(f"警告: 要排除的薄片 (rock_class={rock_class}, rock_id={rock_id}, slide_id={slide_id}) 不存在于数据中")
    print(f"排除后剩余薄片数: {len(df_slides)} (原 {initial_count})")
    return df_slides


def stratified_split_by_slide(df_slides):
    """
    在薄片级别进行分层抽样划分
    返回三个 DataFrame（train, val, test），每个 DataFrame 包含薄片的原始行
    """
    # 准备特征 X 和分层标签 y
    X = df_slides[["slide_key", "rock_class"]]
    y = df_slides["rock_class"]

    # 第一步：分出 test 集（10%）
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        stratify=y,
        random_state=RANDOM_SEED
    )

    # 第二步：从剩余数据中分出 val 集（占剩余数据的比例 = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)）
    # 剩余比例 = 0.9，val 在剩余中的目标比例 = 0.1 / 0.9 ≈ 0.1111
    relative_val_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val_ratio,
        stratify=y_temp,
        random_state=RANDOM_SEED
    )

    # 根据 slide_key 提取原始 DataFrame 行
    train_slides = df_slides[df_slides["slide_key"].isin(X_train["slide_key"])]
    val_slides = df_slides[df_slides["slide_key"].isin(X_val["slide_key"])]
    test_slides = df_slides[df_slides["slide_key"].isin(X_test["slide_key"])]

    print(f"\n薄片划分结果:")
    print(f"  train: {len(train_slides)} 个薄片")
    print(f"  val:   {len(val_slides)} 个薄片")
    print(f"  test:  {len(test_slides)} 个薄片")
    print(f"  合计:  {len(train_slides)+len(val_slides)+len(test_slides)} 个薄片")

    return train_slides, val_slides, test_slides


def expand_to_images(df_slides_split, df_images):
    """
    根据薄片集合，从图像级 DataFrame 中筛选出属于这些薄片的所有图像行
    """
    slide_keys = set(df_slides_split["slide_key"])
    filtered = df_images[df_images["slide_key"].isin(slide_keys)].copy()
    # 移除临时辅助列 slide_key
    filtered.drop(columns=["slide_key"], inplace=True)
    return filtered


def save_split(df_images_split, filename):
    """保存图像级划分结果到 CSV"""
    # 确保输出列都存在
    available_cols = [col for col in OUTPUT_COLUMNS if col in df_images_split.columns]
    out_df = df_images_split[available_cols].copy()
    output_path = OUTPUT_DIR / filename
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已保存: {output_path} ({len(out_df)} 张图像)")
    return out_df


def print_summary(train_df, val_df, test_df, train_slides, val_slides, test_slides):
    """打印统计摘要表格"""
    # 统计每个集合中的薄片数和图像数
    stats = []
    for name, img_df, slide_df in [("train", train_df, train_slides),
                                    ("val",   val_df,   val_slides),
                                    ("test",  test_df,  test_slides)]:
        slides_count = len(slide_df)
        images_count = len(img_df)
        # 按 rock_class 统计图像数量
        class_counts = img_df["rock_class"].value_counts()
        ign = class_counts.get("igneous", 0)
        meta = class_counts.get("metamorphic", 0)
        sed = class_counts.get("sedimentary", 0)
        stats.append([name, slides_count, images_count, ign, meta, sed])

    # 打印表头
    print("\n" + "=" * 70)
    print(f"{'Split':<8} | {'Slides':<6} | {'Images':<6} | {'igneous':<8} | {'metamorphic':<11} | {'sedimentary':<11}")
    print("-" * 70)
    for row in stats:
        print(f"{row[0]:<8} | {row[1]:<6} | {row[2]:<6} | {row[3]:<8} | {row[4]:<11} | {row[5]:<11}")
    print("=" * 70)


def main():
    print("开始加载数据...")
    # 1. 加载薄片摘要
    df_slides = load_slide_summary()
    print(f"原始薄片总数: {len(df_slides)}")

    # 2. 排除异常薄片
    df_slides = exclude_abnormal_slides(df_slides, EXCLUDED_SLIDES)

    # 3. 分层抽样划分薄片
    train_slides, val_slides, test_slides = stratified_split_by_slide(df_slides)

    # 4. 加载图像摘要
    df_images = load_dataset_summary()
    print(f"\n图像总数: {len(df_images)}")

    # 5. 根据薄片集合展开图像
    train_images = expand_to_images(train_slides, df_images)
    val_images = expand_to_images(val_slides, df_images)
    test_images = expand_to_images(test_slides, df_images)

    # 6. 保存划分结果
    print("\n保存划分文件...")
    train_out = save_split(train_images, "train.csv")
    val_out = save_split(val_images, "val.csv")
    test_out = save_split(test_images, "test.csv")

    # 7. 打印统计摘要
    print_summary(train_out, val_out, test_out, train_slides, val_slides, test_slides)

    # 可选：验证图像总数是否匹配
    total_images = len(train_images) + len(val_images) + len(test_images)
    print(f"\n验证: 划分图像总数 = {total_images}，原始图像总数 = {len(df_images)}")
    if total_images != len(df_images):
        print("警告: 划分图像总数与原始图像总数不一致，请检查排除逻辑或薄片匹配情况")


if __name__ == "__main__":
    main()