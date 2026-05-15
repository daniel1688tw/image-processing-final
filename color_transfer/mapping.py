"""色彩映射策略：色差控制 + 色彩一致性維持（內部+外部）。

對應 technology.md §四。輸入兩張影像各自的 Palette，輸出來源每個峰值
（Lab）對應到的目標 Lab 值。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .palette import Palette


class ColorMapper:
    def __init__(self, neighbor_k: int = 3) -> None:
        # 4.3.2 外部連續性的鄰域大小
        self.neighbor_k = neighbor_k

    def build_mapping(self, src: Palette, ref: Palette) -> np.ndarray:
        """回傳 (k_src, 3) 的 Lab 映射目標，依序對應 src.centers。"""
        k_src = src.centers.shape[0]
        if k_src == 0 or ref.centers.shape[0] == 0:
            return src.centers.copy()

        # 4.2 色差控制：每個來源峰值 -> 參考最近鄰
        ref_tree = cKDTree(ref.centers)
        _, nearest_ref = ref_tree.query(src.centers, k=1)

        # 4.3.1 內部一致性：同一參考峰值 j 被多個來源峰值佔用時，只保留來源像素數最多者
        mapping = np.full((k_src, 3), np.nan, dtype=np.float32)
        resolved = np.zeros(k_src, dtype=bool)

        # group source peaks by their nearest ref index
        for ref_j in np.unique(nearest_ref):
            members = np.where(nearest_ref == ref_j)[0]
            winner = members[np.argmax(src.counts[members])]
            mapping[winner] = ref.centers[ref_j]
            resolved[winner] = True
            # 其餘 members 留待外部連續性處理

        # 4.3.2 外部連續性：未解決的峰值用已解決峰值的距離倒數加權
        unresolved = np.where(~resolved)[0]
        resolved_idx = np.where(resolved)[0]
        if unresolved.size > 0 and resolved_idx.size > 0:
            resolved_centers = src.centers[resolved_idx]
            tree = cKDTree(resolved_centers)
            k_neigh = min(self.neighbor_k, resolved_centers.shape[0])
            dists, idx = tree.query(src.centers[unresolved], k=k_neigh)
            if k_neigh == 1:
                dists = dists[:, None]
                idx = idx[:, None]

            # 距離倒數權重；若有完全重合，給該鄰居全部權重
            eps = 1e-6
            inv = 1.0 / (dists + eps)
            weights = inv / inv.sum(axis=1, keepdims=True)

            mapped_neighbors = mapping[resolved_idx][idx]  # (U, k_neigh, 3)
            mapping[unresolved] = (weights[..., None] * mapped_neighbors).sum(axis=1)
        elif unresolved.size > 0:
            # 找不到任何 resolved（理論上 unique 至少給一個，不會到此）
            mapping[unresolved] = src.centers[unresolved]

        return mapping
