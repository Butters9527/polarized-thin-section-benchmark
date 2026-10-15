"""torchvision transform 工厂函数：训练增强 / 验证评估两套流程。"""

from torchvision import transforms


# ImageNet 均值与标准差
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transform() -> transforms.Compose:
    """返回训练集 transform。

    流程：Resize(256) → RandomCrop(224) → 随机翻转/旋转 → ToTensor → ImageNet 归一化。
    先缩放再裁切，避免在原始大图上做旋转导致速度过慢。
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """返回验证集/测试集 transform。

    流程：Resize(256) → CenterCrop(224) → ToTensor → ImageNet 归一化。不含随机增强。
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
