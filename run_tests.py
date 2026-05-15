import os
import shutil
import subprocess
from pathlib import Path

def create_tests():
    # 1. 取得 VOC2012 圖片列表
    voc_dir = Path("VOC2012_test/JPEGImages")
    if not voc_dir.exists():
        print(f"錯誤：找不到資料夾 {voc_dir.absolute()}")
        return

    # 取得前 10 張圖片
    images = sorted(list(voc_dir.glob("*.jpg")))
    if len(images) < 10:
        print(f"警告：圖片數量不足，僅找到 {len(images)} 張。將儘量配對。")
    
    num_tests = 5
    if len(images) < num_tests * 2:
        num_tests = len(images) // 2

    print(f"準備執行 {num_tests} 組測試...")

    for i in range(num_tests):
        test_num = i + 1
        test_folder = Path(f"test_{test_num}")
        test_folder.mkdir(exist_ok=True)

        src_path = images[i * 2]
        ref_path = images[i * 2 + 1]

        # 複製並命名為來源與參考圖
        shutil.copy(src_path, test_folder / "source.jpg")
        shutil.copy(ref_path, test_folder / "reference.jpg")

        # 準備輸出路徑
        out_path = test_folder / "result.jpg"
        
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
