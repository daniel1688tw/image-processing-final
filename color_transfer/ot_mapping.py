"""Optimal Transport-based palette matching.

Drop-in replacement for ColorMapper.build_mapping() that solves a
mass-conserving transport plan instead of nearest-neighbour + ad-hoc
consistency patches (Eqs. 6–10 in Lv & Zhang 2024).

Three variants
--------------
'emd'        – exact OT (Earth Mover's Distance via network simplex).
               Eliminates many-to-one collapses without any repair step.
'sinkhorn'   – entropy-regularised OT.  Soft assignments handle
               concentrated reference distributions more gracefully.
'unbalanced' – unbalanced OT.  Source colors with no good reference
               counterpart are partially left at their original values
               rather than forcibly transported.

Dependencies
------------
Required : POT (pip install POT)
Optional : colour-science (pip install colour-science)
           If present, uses CIEDE2000 (ΔE₀₀) as the transport cost,
           which is perceptually uniform.  Falls back to squared
           Euclidean Lab distance otherwise.
"""

from __future__ import annotations

import numpy as np

from .palette import Palette


def _cost_matrix(
    src_centers: np.ndarray,
    ref_centers: np.ndarray,
    use_delta_e: bool = True,
) -> np.ndarray:
    """Return (K_s, K_r) float64 cost matrix.

    Tries CIEDE2000 first; falls back to squared Euclidean if
    colour-science is not installed.
    """
    if use_delta_e:
        try:
            import colour
            src_exp = src_centers[:, None, :]   # (K_s, 1, 3)
            ref_exp = ref_centers[None, :, :]   # (1,  K_r, 3)
            de = colour.delta_E(src_exp, ref_exp, method='CIE 2000')  # (K_s, K_r)
            return (de ** 2).astype(np.float64)
        except ImportError:
            pass
    diff = src_centers[:, None, :] - ref_centers[None, :, :]   # (K_s, K_r, 3)
    return (diff ** 2).sum(axis=2).astype(np.float64)


def build_ot_mapping(
    src: Palette,
    ref: Palette,
    variant: str = "emd",
    epsilon: float = 0.05,
    tau: float = 0.1,
    use_delta_e: bool = True,
) -> np.ndarray:
    """Return (k_src, 3) Lab mapping targets for each source palette entry.

    Parameters
    ----------
    src, ref    : Palette objects produced by PaletteExtractor
    variant     : 'emd' | 'sinkhorn' | 'unbalanced'
    epsilon     : entropy regularisation for sinkhorn / unbalanced
    tau         : marginal KL penalty (reg_m) for unbalanced OT
    use_delta_e : use CIEDE2000 cost when colour-science is available

    After computing the transport plan T, each source entry i maps to

        new_color(i) = (1 / μ[i]) · Σ_j T[i,j] · ref_centers[j]

    For 'unbalanced', any mass not transported is treated as staying at
    the source color (partial transfer).

    The returned array is a drop-in for ColorMapper.build_mapping() and
    feeds directly into _apply_region_mapping() in transfer.py.
    """
    try:
        import ot as pot
    except ImportError as exc:
        raise ImportError(
            "POT is required for OT-based mapping.  Install with: pip install POT"
        ) from exc

    k_src = src.centers.shape[0]
    k_ref = ref.centers.shape[0]

    if k_src == 0 or k_ref == 0:
        return src.centers.copy()

    # Normalised masses
    mu = src.counts.astype(np.float64)
    mu /= mu.sum()
    nu = ref.counts.astype(np.float64)
    nu /= nu.sum()

    src_f64 = src.centers.astype(np.float64)
    ref_f64 = ref.centers.astype(np.float64)

    C = _cost_matrix(src_f64, ref_f64, use_delta_e)
    # Normalise cost so epsilon is scale-independent (per plan §7)
    C_norm = C / (C.max() + 1e-10)

    if variant == "emd":
        T = pot.emd(mu, nu, C_norm)
    elif variant == "sinkhorn":
        T = pot.sinkhorn(mu, nu, C_norm, reg=epsilon)
    elif variant == "unbalanced":
        T = pot.unbalanced.sinkhorn_unbalanced(mu, nu, C_norm, reg=epsilon, reg_m=tau)
    else:
        raise ValueError(
            f"Unknown OT variant {variant!r}.  Choose 'emd', 'sinkhorn', or 'unbalanced'."
        )

    # Convert transport plan rows → blended target colors
    mapping = np.empty((k_src, 3), dtype=np.float32)
    for i in range(k_src):
        row = T[i]                    # (k_ref,)
        transported = float(row.sum())

        if variant == "unbalanced":
            destroyed = max(mu[i] - transported, 0.0)
            total = transported + destroyed
            if total < 1e-10:
                mapping[i] = src.centers[i]
                continue
            ref_part = (row @ ref_f64) if transported > 1e-10 else np.zeros(3)
            src_part = destroyed * src_f64[i]
            mapping[i] = ((ref_part + src_part) / total).astype(np.float32)
        else:
            if transported < 1e-10:
                mapping[i] = src.centers[i]
            else:
                mapping[i] = ((row @ ref_f64) / transported).astype(np.float32)

    return mapping
