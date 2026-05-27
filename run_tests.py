import os
import shutil
import subprocess
from pathlib import Path

def create_tests():
    # 1. 取得 JPEGImages 圖片列表
    img_dir = Path("JPEGImages")
    if not img_dir.exists():
        print(f"錯誤：找不到資料夾 {img_dir.absolute()}")
        return

    images = sorted(list(img_dir.glob("*.jpg")))
    if len(images) < 2:
        print(f"警告：圖片數量不足，僅找到 {len(images)} 張。")
        return

    pair_count = len(images) // 2
    if len(images) % 2 == 1:
        print(f"提示：圖片總數為奇數 ({len(images)} 張)，將忽略最後一張。")

    sources = images[:pair_count]
    references = images[pair_count:pair_count * 2]

    out_root = Path("outputs")
    sources_dir = out_root / "sources"
    results_dir = out_root / "results"
    sources_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"準備執行 {pair_count} 組測試...")

    for i in range(pair_count):
        test_num = i + 1
        src_path = sources[i]
        ref_path = references[i]

        base_name = f"pair_{test_num:03d}.jpg"
        src_out = sources_dir / base_name
        out_path = results_dir / base_name

        # 複製並命名為來源圖
        shutil.copy(src_path, src_out)
        
        print(f"--- 測試 {test_num} ---")
        print(f"來源: {src_path.name}")
        print(f"參考: {ref_path.name}")
        
        # 執行色彩遷移
        # 這裡預設加上 --no-seg 以確保在沒有安裝 torch 的環境也能運行
        # 如果環境有 torch 且需要語意分割，可以移除 "--no-seg"
        cmd = [
            "python", "main.py",
            "--src", str(src_path),
            "--ref", str(ref_path),
            "--out", str(out_path),
            "--no-seg"
        ]
        
        try: 
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"成功：結果已儲存至 {out_path}")
            else:
                print(f"失敗：{result.stderr}")
        except Exception as e:
            print(f"發生錯誤：{str(e)}")

    print("\n所有測試執行完畢。")

if __name__ == "__main__":
    create_tests()
