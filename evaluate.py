import argparse
from pathlib import Path

import glob
import os
from tqdm import tqdm

import numpy as np
import cv2
from color_transfer.transfer import _bgr_to_lab
from color_transfer.palette import PaletteExtractor
from dataclasses import dataclass

def evaluate_l_consistency(src: np.ndarray, result: np.ndarray, bins=20) -> float:
    src_l = _bgr_to_lab(src)[..., 0] / PaletteExtractor._lab_ranges()[0][1]
    res_l = _bgr_to_lab(result)[..., 0] / PaletteExtractor._lab_ranges()[0][1]
    edges = np.linspace(0, 1, bins + 1)
    img_bins = np.digitize(src_l, edges) - 1

    sum = 0
    valid_bin_count = 0
    for i in range(bins):
        mask = (img_bins == i)
        if np.any(mask):
            sum += res_l[mask].var()
            valid_bin_count += 1
    return sum / valid_bin_count if valid_bin_count > 0 else 0

def evaluate_rgb_consistency(src: np.ndarray, result: np.ndarray, bins=10) -> float:
    edges = np.linspace(0, 255, bins + 1)
    img_bins = np.digitize(src, edges) - 1

    sum = 0
    for c in range(3):
        channel_sum = 0
        valid_bin_count = 0

        img_bins_c = img_bins[..., c]
        result_c = result[..., c]
        for i in range(bins):
            mask = (img_bins_c == i)
            if np.any(mask):
                channel_sum += result_c[mask].var()
                valid_bin_count += 1
        sum += channel_sum / valid_bin_count if valid_bin_count > 0 else 0
    return sum / 3

def evaluate_fading_rates(src: np.ndarray, result: np.ndarray) -> tuple[float, float]:
    src_lab, res_lab = _bgr_to_lab(src)[..., 1:3], _bgr_to_lab(result)[..., 1:3]
    diff_lab = np.clip(abs(src_lab) - abs(res_lab), 0, np.inf)
    return diff_lab[..., 0].mean() / 255, diff_lab[..., 1].mean() / 255


@dataclass
class ColorConsistencyScore:
    """
    l_consistency: L Channel 上的色彩一致性，越高表示越分散 <br>
    rgb_consistency: RGB Channel 上的色彩一致性，越高表示越分散 <br>
    A_fading_rate: Channel A 的褪色程度，越高表示越被弱化 <br>
    B_fading_rate: Channel B 的褪色程度，越高表示越被弱化 <br>
    """
    l_consistency: float
    rgb_consistency: float
    A_fading_rate: float
    B_fading_rate: float

    def __iadd__(self, other: ColorConsistencyScore) -> ColorConsistencyScore:
        self.l_consistency += other.l_consistency
        self.rgb_consistency += other.rgb_consistency
        self.A_fading_rate += other.A_fading_rate
        self.B_fading_rate += other.B_fading_rate
        return self
    
    def __itruediv__(self, divisor: float) -> ColorConsistencyScore:
        self.l_consistency /= divisor
        self.rgb_consistency /= divisor
        self.A_fading_rate /= divisor
        self.B_fading_rate /= divisor
        return self
    
    def __str__(self) -> str:
        return f"Consistency/L={self.l_consistency:.4f}, Consistency/RGB={self.rgb_consistency:.4f}, Fading Rate/A={self.A_fading_rate:.4f}, Fading Rate/B={self.B_fading_rate:.4f}"
    
def evaluate_color_consistency(src: str, res: str, l_bins=20, rgb_bins=10) -> ColorConsistencyScore:
    src = cv2.imread(src)
    res = cv2.imread(res)
    fading_rates = evaluate_fading_rates(src, res)
    return ColorConsistencyScore(
        l_consistency=evaluate_l_consistency(src, res, l_bins),
        rgb_consistency=evaluate_rgb_consistency(src, res, rgb_bins),
        A_fading_rate=fading_rates[0],
        B_fading_rate=fading_rates[1]
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metric', type=str, choices=['brisque', 'clipiqa+', 'cnniqa', 'dbcnn', 'musiq', 'nima', 'ccs'], help='指標名稱，ccs是論文提出的評估方法')
    parser.add_argument('--target', required=True, type=Path, help='目標影像(資料夾或檔案)')
    parser.add_argument('--src', type=Path, help='僅ccs需要，為目標影像(資料夾/檔案)轉換前的原始影像(資料夾/檔案)，若為資料夾則名字須一致')
    parser.add_argument('--num', type=int, default=-1, help='評估前n張影像，預設為全部')
    args = parser.parse_args()

    if args.metric != 'ccs':
        from pyiqa import create_metric
        iqa_model = create_metric(args.metric, metric_mode='FR', device=None)
        avg_score = 0
    else:
        avg_score = ColorConsistencyScore(0, 0, 0, 0)

    target_img_paths = [args.target] if os.path.isfile(args.target) else sorted(glob.glob(os.path.join(args.target, '*')))
    if args.metric == 'ccs':
        original_paths = [args.src] if os.path.isfile(args.src) else [os.path.join(args.src, os.path.basename(p)) for p in target_img_paths]
        target_img_paths = list(zip(target_img_paths, original_paths))

    if args.num > 0 and len(target_img_paths) > args.num:
        target_img_paths = target_img_paths[:args.num]

    for img_path in tqdm(target_img_paths):
        if args.metric == 'ccs':
            score = evaluate_color_consistency(img_path[0], img_path[1])
        else:
            score = iqa_model(str(img_path), None).cpu().item()
        avg_score += score
    avg_score /= len(target_img_paths)

    if args.metric == 'dbcnn':
        avg_score *= 100 # DBCNN 為 [0, 100]，而此module會把該分數normalize到 [0, 1]
    print(f'Average Score: {avg_score}')

if __name__ == '__main__':
    main()