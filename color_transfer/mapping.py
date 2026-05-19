"""色彩映射策略：色差控制 + 色彩一致性維持（內部+外部）。

對應 technology.md §四。輸入兩張影像各自的 Palette，輸出來源每個峰值
（Lab）對應到的目標 Lab 值。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

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
            for m in members:
                if m != winner:
                    offset = src.centers[m] - src.centers[winner]
                    mapping[m] = ref.centers[ref_j] + offset
                    resolved[m] = True

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


class RBFColorTransfer:
    """徑向基底函數 (RBF) 色彩插值與 3D 查找表 (LUT) 加速機制。

    給定來源調色盤 C_src (k, 3) 與目標對應色 C_ref (k, 3)，計算連續映射 f(c):
        f(c) = sum_{j=1}^k (w_j * phi(||c - C_src,j||)) + M * c + t
    其中：
        - phi(r) = exp(-r^2 / (2 * sigma^2)) 為高斯核函數。
        - P(c) = M * c + t 為仿射變換多項式項，以確保仿射穩定性與外部外推的合理性。
    """

    def __init__(
        self,
        c_src: np.ndarray,
        c_ref: np.ndarray,
        sigma: float | None = None,
        grid_size: int = 33,
    ) -> None:
        """初始化 RBF 系統並預先計算 3D LUT。

        c_src: 形狀為 (k, 3) 的來源調色盤中心 (Lab)。
        c_ref: 形狀為 (k, 3) 的目標對應色彩中心 (Lab)。
        sigma: 高斯核的超參數，若為 None 則預設為來源調色盤中心之間的平均距離。
        grid_size: 3D 查找表在各維度上的取樣點數量 (例如 33x33x33)。
        """
        self.c_src = c_src.astype(np.float32)
        self.c_ref = c_ref.astype(np.float32)
        self.k = self.c_src.shape[0]
        self.grid_size = grid_size

        # 若未指定 sigma，設為來源調色盤質心之間的平均 Euclidean 距離
        if sigma is None:
            if self.k > 1:
                # 計算所有質心對之間的距離
                dists = np.linalg.norm(self.c_src[:, None, :] - self.c_src[None, :, :], axis=-1)
                # 取上三角（不含對角線）的均值
                triu_idx = np.triu_indices(self.k, k=1)
                mean_dist = dists[triu_idx].mean()
                self.sigma = float(np.maximum(15.0, mean_dist * 0.8))
            else:
                self.sigma = 20.0
        else:
            self.sigma = sigma

        self._solve_system()
        self._precompute_lut()

    def _solve_system(self) -> None:
        """建構並求解滿足插值與正交約束的 block 矩陣系統。

        矩陣系統：
        [ A    X  1 ] [  w  ]   [ Y ]
        [ X^T  0  0 ] [ M^T ] = [ 0 ]
        [ 1^T  0  0 ] [ t^T ]   [ 0 ]
        其中：
          - A 為 (k, k) 核心矩陣，A_ij = phi(||C_src,i - C_src,j||)
          - X 為 C_src (k, 3)
          - 1 為 (k, 1) 的全 1 向量
          - Y 為 C_ref (k, 3)
        """
        k = self.k
        if k == 0:
            self.w = np.zeros((0, 3), dtype=np.float32)
            self.M = np.zeros((3, 3), dtype=np.float32)
            self.t = np.zeros(3, dtype=np.float32)
            return

        # 核心矩陣 A
        dists = np.linalg.norm(self.c_src[:, None, :] - self.c_src[None, :, :], axis=-1)
        A = np.exp(-(dists ** 2) / (2.0 * self.sigma ** 2))
        A += np.eye(k, dtype=np.float32) * 0.1

        # 建構大區塊矩陣 L (k+4, k+4)
        L = np.zeros((k + 1, k + 1), dtype=np.float32)
        L[:k, :k] = A
        L[:k, k] = 1.0
        L[k, :k] = 1.0

        # 建構右側 B (k+1, 3)
        B = np.zeros((k + 1, 3), dtype=np.float32)
        B[:k] = self.c_ref - self.c_src

        # 求解 L * Z = B
        try:
            # 優先使用標準高效 solver
            Z = np.linalg.solve(L, B)
        except np.linalg.LinAlgError:
            # 若系統奇異（如 k < 4 時），退回使用最小平方法求解
            Z, _, _, _ = np.linalg.lstsq(L, B, rcond=1e-5)

        self.w = Z[:k, :]                  
        self.t = Z[k, :]

    def evaluate_rbf(self, coords: np.ndarray) -> np.ndarray:
        """直接計算一組 Lab 座標的 RBF 連續映射值。

        coords: (N, 3) 的 Lab 座標陣列。
        """
        if self.k == 0:
            return np.zeros_like(coords)

        # 計算輸入點到所有質心的距離: (N, k)
        dists = np.linalg.norm(coords[:, None, :] - self.c_src[None, :, :], axis=-1)
        phi = np.exp(-(dists ** 2) / (2.0 * self.sigma ** 2))

        # RBF 加權和: (N, k) @ (k, 3) -> (N, 3)
        rbf_term = phi @ self.w

        return rbf_term + self.t

    def _precompute_lut(self) -> None:
        """在 3D Lab 空間中預先計算 RBF 的變形場 (LUT)。"""
        # Lab 各通道的有效值邊界範圍
        self.L_range = np.linspace(0.0, 100.0, self.grid_size, dtype=np.float32)
        self.a_range = np.linspace(-128.0, 127.0, self.grid_size, dtype=np.float32)
        self.b_range = np.linspace(-128.0, 127.0, self.grid_size, dtype=np.float32)

        # 建立 3D 網格
        L_grid, a_grid, b_grid = np.meshgrid(self.L_range, self.a_range, self.b_range, indexing="ij")
        grid_coords = np.stack([L_grid.ravel(), a_grid.ravel(), b_grid.ravel()], axis=1)

        # 評估 RBF 映射值
        shift_grid = self.evaluate_rbf(grid_coords)
        transformed_grid = grid_coords + shift_grid

        # 重新整理成 3D LUT (grid_size, grid_size, grid_size, 3)
        self.lut = transformed_grid.reshape(self.grid_size, self.grid_size, self.grid_size, 3)

        # 建立三線性內插器 (RegularGridInterpolator)
        self.interpolator = RegularGridInterpolator(
            points=(self.L_range, self.a_range, self.b_range),
            values=self.lut,
            method="linear",
            bounds_error=False,
            fill_value=None  # 自動外推
        )

    def map_colors(self, pixels: np.ndarray) -> np.ndarray:
        """使用預計算 3D LUT 對像素色彩進行映射，並套用 gamut 色域保護。

        pixels: (N, 3) 的 Lab 座標陣列。
        """
        # 三線性內插
        mapped = self.interpolator(pixels)

        # Gamut Safeguard: 限制 Lab 輸出於合理範圍內，防止外推時溢出
        clamped = np.clip(mapped, [0.0, -128.0, -128.0], [100.0, 127.0, 127.0])
        return clamped

