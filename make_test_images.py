"""產生兩張具有不同色調的合成測試影像，供色彩遷移端對端測試使用。"""

from pathlib import Path

import cv2
import numpy as np


def warm_source(h: int = 384, w: int = 384) -> np.ndarray:
    """暖色調：天空淡橘 + 草地黃綠 + 中央紅色物件。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy = np.linspace(0, 1, h)[:, None]
    sky_top = np.array([180, 200, 235], dtype=np.float32)   # BGR
    sky_bot = np.array([120, 170, 220], dtype=np.float32)
    sky = sky_top * (1 - yy) + sky_bot * yy
    img[: int(h * 0.6)] = sky[: int(h * 0.6)].astype(np.uint8)[:, None].repeat(w, axis=1)

    grass_top = np.array([60, 160, 180], dtype=np.float32)
    grass_bot = np.array([30, 110, 140], dtype=np.float32)
    g_h = h - int(h * 0.6)
    yy2 = np.linspace(0, 1, g_h)[:, None]
    grass = grass_top * (1 - yy2) + grass_bot * yy2
    img[int(h * 0.6) :] = grass.astype(np.uint8)[:, None].repeat(w, axis=1)

    cv2.circle(img, (w // 2, int(h * 0.55)), 70, (40, 60, 200), -1)   # 紅色物件
    cv2.circle(img, (w // 2, int(h * 0.55)), 70, (20, 30, 120), 4)
    cv2.rectangle(img, (40, int(h * 0.7)), (140, int(h * 0.95)), (50, 80, 110), -1)  # 棕屋
    return img


def cool_reference(h: int = 384, w: int = 384) -> np.ndarray:
    """冷色調：深藍天空 + 青綠地 + 中央紫色物件。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy = np.linspace(0, 1, h)[:, None]
    sky_top = np.array([110, 60, 30], dtype=np.float32)     # 深藍
    sky_bot = np.array([180, 120, 70], dtype=np.float32)
    sky = sky_top * (1 - yy) + sky_bot * yy
    img[: int(h * 0.55)] = sky[: int(h * 0.55)].astype(np.uint8)[:, None].repeat(w, axis=1)

    grass_top = np.array([140, 130, 60], dtype=np.float32)
    grass_bot = np.array([110, 90, 30], dtype=np.float32)
    g_h = h - int(h * 0.55)
    yy2 = np.linspace(0, 1, g_h)[:, None]
    grass = grass_top * (1 - yy2) + grass_bot * yy2
    img[int(h * 0.55) :] = grass.astype(np.uint8)[:, None].repeat(w, axis=1)

    cv2.circle(img, (w // 2, int(h * 0.5)), 80, (160, 80, 150), -1)   # 紫色物件
    cv2.circle(img, (w // 2, int(h * 0.5)), 80, (90, 40, 80), 4)
    cv2.rectangle(img, (60, int(h * 0.7)), (160, int(h * 0.95)), (130, 100, 60), -1)
    return img


def main() -> None:
    out_dir = Path(__file__).parent / "test_images"
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "source.png"), warm_source())
    cv2.imwrite(str(out_dir / "reference.png"), cool_reference())
    print(f"已輸出: {out_dir}")


if __name__ == "__main__":
    main()
