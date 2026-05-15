"""前景/背景分離：以 torchvision 的 DeepLabV3 取代論文 DeepLabV3+。

若環境未安裝 PyTorch / torchvision，依論文設計退化為「全部視為前景」。
"""

from __future__ import annotations

import numpy as np

_BACKGROUND_CLASS = 0  # VOC 標準：0 為背景


def _try_load_deeplab():
    try:
        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet101_Weights,
            deeplabv3_resnet101,
        )
    except Exception:  # noqa: BLE001 — 套件不可用就退化
        return None, None, None

    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, weights.transforms(), device


_CACHE: dict = {}


def segment_foreground(image_rgb: np.ndarray) -> np.ndarray:
    """回傳 (H, W) bool 的前景遮罩；無法載入模型時整張視為前景。

    image_rgb: uint8 (H, W, 3)
    """
    if "model" not in _CACHE:
        _CACHE["model"], _CACHE["transform"], _CACHE["device"] = _try_load_deeplab()

    model = _CACHE["model"]
    if model is None:
        return np.ones(image_rgb.shape[:2], dtype=bool)

    import torch
    from PIL import Image

    transform = _CACHE["transform"]
    device = _CACHE["device"]

    pil = Image.fromarray(image_rgb)
    batch = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)["out"][0]
    pred = logits.argmax(0).cpu().numpy()

    # 模型輸入經 resize，需把預測還原到原尺寸
    if pred.shape != image_rgb.shape[:2]:
        from PIL import Image as _Image

        pred_img = _Image.fromarray(pred.astype(np.uint8)).resize(
            (image_rgb.shape[1], image_rgb.shape[0]), resample=_Image.NEAREST
        )
        pred = np.array(pred_img)

    foreground = pred != _BACKGROUND_CLASS
    # 若整張幾乎沒有前景（< 1%），論文設計：全部視為前景
    if foreground.mean() < 0.01:
        return np.ones_like(foreground, dtype=bool)
    return foreground.astype(bool)
