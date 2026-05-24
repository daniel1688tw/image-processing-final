"""命令列入口：

    python main.py --src path/to/source.jpg --ref path/to/reference.jpg --out result.png

選項：
    --no-seg          跳過 DeepLabV3，整張視為前景
    --alpha 0.3       光照混合權重 (§5.2)
    --max-palette 32  調色盤上限 t (§3.3)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from color_transfer import transfer_color
from color_transfer.transfer import TransferConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Palette-based color transfer (Lv & Zhang 2024)")
    p.add_argument("--src", required=True, type=Path, help="來源影像")
    p.add_argument("--ref", required=True, type=Path, help="參考影像")
    p.add_argument("--out", required=True, type=Path, help="輸出影像路徑")
    p.add_argument("--no-seg", action="store_true", help="不使用語意分割")
    p.add_argument("--alpha", type=float, default=0.3, help="L 通道映射權重")
    p.add_argument("--max-palette", type=int, default=32, help="調色盤峰值上限 t")
    p.add_argument("--bins", type=int, default=100, help="直方圖分箱數 z")
    p.add_argument(
        "--ot-variant",
        choices=["emd", "sinkhorn", "unbalanced"],
        default=None,
        help="OT-based palette matching variant (default: baseline nearest-neighbour)",
    )
    p.add_argument("--ot-epsilon", type=float, default=0.05, help="Sinkhorn/unbalanced entropy reg ε")
    p.add_argument("--ot-tau", type=float, default=0.1, help="Unbalanced OT marginal penalty τ")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = cv2.imread(str(args.src))
    ref = cv2.imread(str(args.ref))
    if src is None:
        raise FileNotFoundError(f"無法讀取來源影像: {args.src}")
    if ref is None:
        raise FileNotFoundError(f"無法讀取參考影像: {args.ref}")

    cfg = TransferConfig(
        bins=args.bins,
        max_palette_size=args.max_palette,
        lighting_alpha=args.alpha,
        use_segmentation=not args.no_seg,
        ot_variant=args.ot_variant,
        ot_epsilon=args.ot_epsilon,
        ot_tau=args.ot_tau,
    )
    result = transfer_color(src, ref, cfg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), result)
    print(f"已輸出: {args.out}")


if __name__ == "__main__":
    main()
