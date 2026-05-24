# 影像處理專題：基於調色盤的色彩遷移 (Palette-based Color Transfer)

本專題實現了一種基於調色盤 (Palette) 的色彩遷移演算法，結合語意分割、調色盤萃取、最佳傳輸匹配、色差控制與光照優化，能夠將參考影像的色調精準地遷移至來源影像中。另外提供四種調色盤匹配策略（基線最近鄰、EMD、Sinkhorn、Unbalanced OT）的視覺化比較工具。

論文來源：Chenlei Lv, Dan Zhang, *Palette-based Color Transfer between Images*, arXiv:2405.08263, 2024.

---

## 專案結構

```
.
├── color_transfer/          # 核心套件
│   ├── __init__.py          # 套件入口，匯出 transfer_color 等公開介面
│   ├── palette.py           # 調色盤萃取（Lab 直方圖峰值搜尋 + kd-tree 合併）
│   ├── mapping.py           # 色彩映射（色差控制 + 內部一致性 + 外部連續性）
│   ├── ot_mapping.py        # 最佳傳輸調色盤匹配（EMD / Sinkhorn / Unbalanced OT）
│   ├── segmentation.py      # 語意分割（DeepLabV3-ResNet101 前景/背景分離）
│   └── transfer.py          # 主 pipeline（串聯以上模組，輸出 BGR uint8 結果）
├── app.py                   # Gradio 網頁介面（上傳圖片即可使用）
├── main.py                  # 命令列入口（單張影像處理）
├── case_study.py            # 四種方法並排比較視覺化（Lab gamut、SSIM、ΔE₀₀）
├── run_tests.py             # 批量測試腳本（從 VOC2012_test 自動生成 5 組測試）
├── make_test_images.py      # 合成測試影像產生器
├── requirements.txt         # Python 套件依賴清單
└── technology.md            # 技術架構詳解文件
```

---

## 環境需求

- Python **3.9 以上**（建議使用 venv 隔離套件）
- 主要依賴：`numpy`, `opencv-python`, `scipy`, `gradio`
- 最佳傳輸依賴：`POT>=0.9`, `colour-science>=0.4`, `scikit-image>=0.21`
- 語意分割依賴（可選）：`torch`, `torchvision`, `pillow`

---

## 安裝步驟

```bash
# 1. 建立虛擬環境
python3 -m venv .venv

# 2. 啟動虛擬環境
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. 安裝所有套件
pip install -r requirements.txt
```

---

## 執行方式

### 方式一：Gradio 網頁介面（推薦）

```bash
source .venv/bin/activate
python app.py
```

啟動後在瀏覽器開啟終端機顯示的網址（預設 `http://127.0.0.1:7860`），上傳來源影像與參考影像後點擊 **Run Transfer** 即可。

---

### 方式二：命令列（單張影像）

```bash
python main.py --src path/to/source.jpg --ref path/to/reference.jpg --out result.png
```

**可用選項：**

| 選項 | 預設值 | 說明 |
|------|--------|------|
| `--no-seg` | — | 關閉語意分割，整張影像視為前景（無 PyTorch 環境時使用） |
| `--alpha 0.3` | 0.3 | L 通道光照混合權重（§5.2），越大則結果越接近參考影像的光照 |
| `--max-palette 32` | 32 | 調色盤峰值數量上限 t |
| `--bins 100` | 100 | Lab 各通道直方圖分箱數 z |
| `--ot-variant` | — | OT 匹配策略：`emd` / `sinkhorn` / `unbalanced`（省略則使用基線最近鄰） |
| `--ot-epsilon 0.05` | 0.05 | Sinkhorn / Unbalanced OT 的熵正規化係數 ε |
| `--ot-tau 0.1` | 0.1 | Unbalanced OT 的邊際懲罰係數 τ |

**使用 OT 匹配的範例：**

```bash
# 精確 EMD（消除多對一坍塌）
python main.py --src source.jpg --ref reference.jpg --out result_emd.png --ot-variant emd

# Sinkhorn（熵正規化，適合色彩集中的參考影像）
python main.py --src source.jpg --ref reference.jpg --out result_sk.png --ot-variant sinkhorn --ot-epsilon 0.05

# Unbalanced OT（允許部分傳輸，保留來源中無對應色彩的條目）
python main.py --src source.jpg --ref reference.jpg --out result_ub.png --ot-variant unbalanced --ot-epsilon 0.05 --ot-tau 0.1
```

---

### 方式三：Case Study 四方法比較

對同一組來源/參考影像執行全部四種匹配策略，並輸出包含以下內容的並排視覺化圖表：

- 來源、參考、四種結果影像
- 各結果的 Lab gamut 散點圖（a*、b* 分布，含調色盤中心與映射箭頭）
- SSIM 與加權 ΔE₀₀ 指標列

```bash
python case_study.py --src test_0/src.png --ref test_0/ref.jpg --out-dir test_0
```

輸出至 `<out-dir>/case_study.png`。

---

### 方式四：批量測試

從 `VOC2012_test/JPEGImages/` 自動取前 10 張圖配對為 5 組測試，結果輸出至 `test_1/` ~ `test_5/`。

```bash
python run_tests.py
```

---

### 方式五：產生合成測試影像

產生一組暖色調來源影像與冷色調參考影像，輸出至 `test_images/`。

```bash
python make_test_images.py
```

---

## 技術流程概述

1. **Lab 色彩空間轉換**：在感知均勻的空間中處理，亮度（L）與色彩（a, b）解耦。
2. **語意分割**：使用 DeepLabV3-ResNet101 分離前景與背景，兩者分別建立調色盤與映射。
3. **調色盤萃取**：對各 Lab 通道分別進行 1D 直方圖峰值搜尋，取三通道峰值的 Cartesian 組合為候選中心，再以 kd-tree 分配像素，保留票數最多的前 t 個作為最終調色盤。
4. **調色盤匹配**（四種策略）：
   - **基線（最近鄰）**：對每個來源調色盤中心找最近的參考中心，並以內部一致性與外部連續性修正多對一映射。
   - **EMD**：精確最佳傳輸（網路單純形法），保證質量守恆，消除多對一坍塌。
   - **Sinkhorn**：熵正規化 OT，對高度集中的參考調色盤產生更平滑的指派。
   - **Unbalanced OT**：允許部分傳輸；距離過遠的來源色彩保留原色，適合來源/參考色域差異極大的情境。
   - 成本矩陣以 CIEDE2000（ΔE₀₀）計算，感知精度優於歐氏距離。
5. **像素轉換（軟指派）**：每個像素查詢最近 k 個調色盤中心，以距離倒數加權混合映射結果，消除梯度區域的色帶斷層。新色彩 = 加權映射目標 + 原像素與加權中心的偏差（保留細節）。
6. **光照優化**：a、b 通道直接採用映射結果；L 通道以加權混合 `(1-α)·L_src + α·L_mapped`（預設 α=0.3）保留原始光照結構。

詳細說明請參閱 [`technology.md`](technology.md)。
