"""調色盤分群：以 Lab 直方圖峰值搜尋為核心的自適應分群法。

對應 technology.md §三。流程：
  1. 各通道分箱直方圖（z=100）
  2. 局部峰值搜尋（半徑 r=3，閾值 b_min=30）
  3. 三通道候選做 Cartesian 組合 -> kd-tree 分配像素 -> 取像素數最多的前 t=32 個
  4. 像素以最近峰值重新分類，平均色為調色盤項目
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Palette:
    """單張影像的調色盤輸出。

    centers: (k, 3) Lab 空間下的調色盤色彩中心。
    counts:  (k,)   每個中心對應的像素數。
    labels:  (H, W) 每個像素被指派到的中心索引。
    """

    centers: np.ndarray
    counts: np.ndarray
    labels: np.ndarray


class PaletteExtractor:
    def __init__(
        self,
        bins: int = 100,
        peak_radius: int = 3,
        peak_min_count: int = 30,
        max_palette_size: int = 32,
    ) -> None:
        self.bins = bins
        self.peak_radius = peak_radius
        self.peak_min_count = peak_min_count
        self.max_palette_size = max_palette_size

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _lab_ranges() -> tuple[tuple[float, float], ...]:
        # OpenCV float32 Lab: L in [0,100], a/b roughly in [-128,127]
        return ((0.0, 100.0), (-128.0, 127.0), (-128.0, 127.0))

    def _channel_histogram(self, values: np.ndarray, lo: float, hi: float):
        hist, edges = np.histogram(values, bins=self.bins, range=(lo, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        return hist, centers

    def _find_peaks_1d(self, hist: np.ndarray, centers: np.ndarray) -> np.ndarray:
        r = self.peak_radius
        peaks: list[float] = []
        n = len(hist)
        for i in range(n):
            if hist[i] <= self.peak_min_count:
                continue
            lo = max(0, i - r)
            hi = min(n, i + r + 1)
            if hist[i] == hist[lo:hi].max():
                peaks.append(centers[i])
        return np.asarray(peaks, dtype=np.float32)

    # ------------------------------------------------------------------ main
    def extract(self, image_lab: np.ndarray, mask: np.ndarray | None = None) -> Palette:
        """擷取一張 Lab 影像的調色盤。

        image_lab: (H, W, 3) float32, OpenCV 的 Lab 範圍。
        mask:      (H, W) bool，True 表示納入分析的像素；None 為全採用。
        回傳的 labels 對 mask 之外的像素填 -1。
        """
        if image_lab.dtype != np.float32:
            image_lab = image_lab.astype(np.float32)
        h, w, _ = image_lab.shape

        if mask is None:
            mask = np.ones((h, w), dtype=bool)
        flat_pixels = image_lab[mask]  # (M, 3)
        if flat_pixels.shape[0] == 0:
            return Palette(
                centers=np.zeros((0, 3), dtype=np.float32),
                counts=np.zeros((0,), dtype=np.int64),
                labels=np.full((h, w), -1, dtype=np.int32),
            )

        # 1) per-channel histograms + 1D peaks
        ranges = self._lab_ranges()
        per_channel_peaks: list[np.ndarray] = []
        for c in range(3):
            lo, hi = ranges[c]
            hist, centers = self._channel_histogram(flat_pixels[:, c], lo, hi)
            peaks = self._find_peaks_1d(hist, centers)
            if peaks.size == 0:
                # 沒有滿足條件的峰值就退而求其次：取整體平均
                peaks = np.array([float(flat_pixels[:, c].mean())], dtype=np.float32)
            per_channel_peaks.append(peaks)

        # 2) Cartesian 組合成 3D 候選峰值
        Lp, Ap, Bp = per_channel_peaks
        gl, ga, gb = np.meshgrid(Lp, Ap, Bp, indexing="ij")
        candidates = np.stack([gl.ravel(), ga.ravel(), gb.ravel()], axis=1).astype(np.float32)

        # 3) 以 kd-tree 將像素分配到候選峰值，統計票數
        tree = cKDTree(candidates)
        _, idx = tree.query(flat_pixels, k=1)
        votes = np.bincount(idx, minlength=candidates.shape[0])

        # 取前 t 名（票數需 > 0）
        t = min(self.max_palette_size, int((votes > 0).sum()))
        if t == 0:
            t = 1
        top = np.argsort(votes)[::-1][:t]
        top_centers = candidates[top]

        # 4) 以 top 候選重新做最近鄰分配，並用該分群的像素均值作為最終調色盤
        tree_top = cKDTree(top_centers)
        _, final_idx = tree_top.query(flat_pixels, k=1)

        final_centers = np.zeros((t, 3), dtype=np.float32)
        counts = np.zeros((t,), dtype=np.int64)
        for k in range(t):
            sel = flat_pixels[final_idx == k]
            counts[k] = sel.shape[0]
            if sel.shape[0] > 0:
                final_centers[k] = sel.mean(axis=0)
            else:
                final_centers[k] = top_centers[k]

        # 移除空群
        keep = counts > 0
        final_centers = final_centers[keep]
        counts = counts[keep]
        # final_idx 需要重新映射到壓縮後的索引
        remap = -np.ones(t, dtype=np.int32)
        remap[np.where(keep)[0]] = np.arange(int(keep.sum()))
        final_idx = remap[final_idx]

        labels = np.full((h, w), -1, dtype=np.int32)
        labels[mask] = final_idx
        return Palette(centers=final_centers, counts=counts, labels=labels)
