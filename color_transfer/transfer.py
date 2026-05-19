"""主流程：把調色盤分群、分割對應、色彩映射、光照優化串成一條 pipeline。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .mapping import ColorMapper, RBFColorTransfer
from .palette import Palette, PaletteExtractor
from .segmentation import segment_foreground


@dataclass
class TransferConfig:
    bins: int = 100
    peak_radius: int = 3
    peak_min_count: int = 30
    max_palette_size: int = 32
    lighting_alpha: float = 0.3        # §5.2 預設 0.3
    use_segmentation: bool = True       # 失敗或關閉時整張視為前景
    neighbor_k: int = 3                 # §4.3.2 鄰域大小
    rbf_sigma: float | None = None      # RBF 高斯核超參數 sigma
    lut_grid_size: int = 33             # 3D LUT 網格大小


def _bgr_to_lab(image_bgr: np.ndarray) -> np.ndarray:
    rgb_f = image_bgr[..., ::-1].astype(np.float32) / 255.0
    return cv2.cvtColor(rgb_f, cv2.COLOR_RGB2LAB)


def _lab_to_bgr(image_lab: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    rgb = np.clip(rgb, 0.0, 1.0)
    bgr = (rgb[..., ::-1] * 255.0).round().astype(np.uint8)
    return bgr


def _apply_region_mapping(
    src_lab: np.ndarray,
    src_palette: Palette,
    mapping: np.ndarray,
    region_mask: np.ndarray,
    out_lab: np.ndarray,
    sigma: float | None = None,
    grid_size: int = 33,
) -> None:
    """對 region_mask 內的像素套用 RBF-LUT 連續色彩對應。out_lab 會被原地更新。"""
    if src_palette.centers.shape[0] == 0 or mapping.shape[0] == 0:
        return
    # 對應像素的調色盤索引
    labels = src_palette.labels[region_mask]      # (M,)
    valid = labels >= 0
    if not valid.any():
        return
    pixel_lab = src_lab[region_mask][valid]        # (V, 3)

    # --- 舊有的 Nearest Neighbor 離散映射邏輯 (已註解以利代碼完整性) ---
    # centers = src_palette.centers[labels[valid]]   # (V, 3) b_i
    # targets = mapping[labels[valid]]                # (V, 3) f(b_i)
    # # f(p_i) = f(b_i) + (pixel_lab - b_i)
    # new_lab = targets + (pixel_lab - centers)
    # -------------------------------------------------------------

    # --- 新有的 RBF-LUT 連續插值與色域保護映射 ---
    rbf_transfer = RBFColorTransfer(src_palette.centers, mapping, sigma=sigma, grid_size=grid_size)
    new_lab = rbf_transfer.map_colors(pixel_lab)
    # ---------------------------------------------

    # 寫回：region_mask 內 valid 的位置
    region_indices = np.where(region_mask)
    rows = region_indices[0][valid]
    cols = region_indices[1][valid]
    out_lab[rows, cols] = new_lab


def transfer_color(
    src_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    config: TransferConfig | None = None,
) -> np.ndarray:
    """執行論文流程，回傳 BGR uint8 結果。"""
    cfg = config or TransferConfig()

    src_lab = _bgr_to_lab(src_bgr)
    ref_lab = _bgr_to_lab(ref_bgr)

    # 1) 分割對應：取得前景/背景 mask
    if cfg.use_segmentation:
        src_fg = segment_foreground(src_bgr[..., ::-1])
        ref_fg = segment_foreground(ref_bgr[..., ::-1])
    else:
        src_fg = np.ones(src_bgr.shape[:2], dtype=bool)
        ref_fg = np.ones(ref_bgr.shape[:2], dtype=bool)

    extractor = PaletteExtractor(
        bins=cfg.bins,
        peak_radius=cfg.peak_radius,
        peak_min_count=cfg.peak_min_count,
        max_palette_size=cfg.max_palette_size,
    )
    mapper = ColorMapper(neighbor_k=cfg.neighbor_k)

    # 2) 為每個區域萃取調色盤；建立 f_full Lab 結果（先複製來源）
    f_full_lab = src_lab.copy()

    def process_region(src_mask: np.ndarray, ref_mask: np.ndarray) -> None:
        if not src_mask.any():
            return
        src_pal = extractor.extract(src_lab, src_mask)
        # 參考側即使沒有對應前景，也要有調色盤；若空就取整張
        ref_eff_mask = ref_mask if ref_mask.any() else np.ones_like(ref_mask, dtype=bool)
        ref_pal = extractor.extract(ref_lab, ref_eff_mask)
        if ref_pal.centers.shape[0] == 0:
            return
        mapping = mapper.build_mapping(src_pal, ref_pal)
        _apply_region_mapping(
            src_lab,
            src_pal,
            mapping,
            src_mask,
            f_full_lab,
            sigma=cfg.rbf_sigma,
            grid_size=cfg.lut_grid_size,
        )

    process_region(src_fg, ref_fg)
    src_bg = ~src_fg
    ref_bg = ~ref_fg
    if src_bg.any():
        process_region(src_bg, ref_bg)

    # 3) §4.4 註：a, b 通道直接採用 f_full；L 通道走 §5.2 加權
    out_lab = src_lab.copy()
    out_lab[..., 1] = f_full_lab[..., 1]
    out_lab[..., 2] = f_full_lab[..., 2]
    alpha = cfg.lighting_alpha
    out_lab[..., 0] = (1.0 - alpha) * src_lab[..., 0] + alpha * f_full_lab[..., 0]

    return _lab_to_bgr(out_lab)
