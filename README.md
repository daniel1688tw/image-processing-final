# 影像處理專題：基於調色盤的色彩遷移 (Palette-based Color Transfer)

本專題實現了一種基於調色盤 (Palette) 的色彩遷移演算法，結合了語意分割、調色盤萃取、色差控制與光照優化等技術，能夠將參考影像的色調精準地遷移至來源影像中。

---

## 專案結構與檔案說明

### 核心模組 (`color_transfer/`)
- **`__init__.py`**: 套件入口，匯出核心函式。
- **`palette.py`**: **調色盤萃取**。將影像轉換至 Lab 色彩空間，並透過三維直方圖與峰值偵測演算法提取代表性色調。
- **`mapping.py`**: **色彩映射邏輯**。根據來源與參考影像的調色盤，計算色調對應關係，並處理內部一致性與外部連續性。
- **`segmentation.py`**: **語意分割**。使用 DeepLabV3 模型進行前景與背景分離，確保物體與背景的色彩能分別遷移。
- **`transfer.py`**: **核心 Pipeline**。串聯分割、萃取、映射與最終的 Lab 通道合成與光照優化 (§5.2)。

### 工具與入口腳本
- **`main.py`**: 命令列入口。使用者可以透過此腳本對單一圖片進行色彩遷移。
- **`run_tests.py`**: 自動化批量測試腳本。從 `VOC2012_test` 資料集中挑選圖片並生成五組測試結果。
- **`make_test_images.py`**: 測試影像生成工具。產生具有明顯色調差異的合成影像，用於驗證演算法穩定性。
- **`requirements.txt`**: 列出專案所需的 Python 套件（numpy, opencv, scipy, torch, torchvision, pillow）。


---



# 2. 安裝必要套件 (若虛擬環境尚未安裝)
pip install -r requirements.txt
```

---

## 執行指令說明

### 1. 單張影像色彩遷移
使用 `main.py` 進行處理。

```powershell
python main.py --src path/to/source.jpg --ref path/to/reference.jpg --out result.png
```

**常用選項：**
- `--no-seg`: 關閉語意分割（若環境未安裝 PyTorch 或想加快速度時使用）。
- `--alpha 0.3`: 調整光照混合權重（預設 0.3，數值越大則結果圖的光照越接近參考圖）。
- `--max-palette 32`: 調色盤峰值上限。

### 2. 執行批量測試
執行我為您撰寫的批量腳本，將自動生成 5 組測試資料夾（test_1 ~ test_5）。

```powershell
python run_tests.py
```

### 3. 產生合成測試圖
```powershell
python make_test_images.py
```
執行後會在 `test_images/` 資料夾中產生 `source.png` 與 `reference.png`。

---

## 技術細節
本專案參考了相關學術論文（如 Lv & Zhang 2024），核心流程包含：
1. **Lab 空間轉換**：在感知均勻的色彩空間中處理。
2. **三維直方圖峰值偵測**：自動尋找影像中的主導色彩。
3. **區塊一致性映射**：結合語意分割，避免前景與背景色彩混淆。
4. **光照保持**：僅在 L 通道進行權重混合，保留原始影像的明暗結構。
