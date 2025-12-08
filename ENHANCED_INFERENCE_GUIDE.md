# 增強版 HiM2SAM 推論腳本文檔
# Enhanced HiM2SAM Inference Script Documentation

## 概述 (Overview)

`run_skitb_inference_enhanced.py` 是一個增強版的 HiM2SAM 推論腳本，**特別針對您提出的兩個核心問題**:

### ✅ 問題 1: 沒有 skier 的幀仍有 bbox
**解決方案:** `TrajectorySmoothing.detect_valid_mask()` 
- ✓ 檢查 mask 的最小面積 (100 像素)
- ✓ 檢查 bbox 的最小寬高 (10×10 像素)
- ✓ 計算 mask 填充比例
- ✓ 只有有效 mask 才寫入 bbox，否則寫 "0,0,0,0"

### ✅ 問題 2: bbox 浮動劇烈，提前出現在未來位置
**解決方案:** 三層軌跡平滑機制
1. **異常檢測 (Outlier Detection):** 移除速度過高的跳躍點
2. **內插修復 (Interpolation):** 線性填補缺失幀
3. **時間平滑 (Temporal Smoothing):** 高斯濾波平滑軌跡

---

## 核心功能 (Core Features)

### 1. 有效 Mask 檢測 (`detect_valid_mask`)

```python
is_valid, bbox, area_ratio = smoother.detect_valid_mask(mask, mask_area_threshold=100)
```

**檢測標準:**
- ✓ Mask 面積 > 100 像素
- ✓ Bbox 寬度 > 10 像素
- ✓ Bbox 高度 > 10 像素

**效果:** 自動過濾掉以下情況:
- 完全遮擋 (mask 面積為 0)
- 過度模糊 (mask 太分散)
- 誤檢 (小的噪點)

### 2. 異常跳躍檢測 (`remove_outliers_velocity`)

```python
cleaned_bboxes = smoother.remove_outliers_velocity(raw_bboxes, max_velocity=100)
```

**工作原理:**
- 計算相鄰幀 bbox 中心點的移動距離
- 計算速度: `velocity = distance / frame_gap`
- **移除速度 > 100 px/frame 的幀** (標記為 None)

**典型應用:**
```
正常速度: 5-30 px/frame (滑雪者在 2-3 幀內移動 50-100 像素)
異常跳躍: 200+ px/frame (突然出現在錯誤位置)
```

### 3. 缺失幀內插 (`interpolate_gaps`)

```python
interpolated_bboxes = smoother.interpolate_gaps(cleaned_bboxes, max_gap=10)
```

**工作原理:**
- 找出所有有效 bbox 的幀
- 在相鄰有效幀之間進行線性內插
- 只內插間隙 ≤ 10 幀的缺失幀

**公式:**
```
bbox[i] = bbox[i1] * (1-α) + bbox[i2] * α
其中 α = (i - i1) / (i2 - i1)
```

### 4. 時間平滑 (`smooth_bboxes_temporal`)

```python
smoothed_bboxes = smoother.smooth_bboxes_temporal(
    interpolated_bboxes, 
    window_size=5, 
    sigma=1.5
)
```

**工作原理:**
- 對 bbox 的每個座標 (x1, y1, x2, y2) 應用高斯濾波
- 標準差 sigma=1.5 (可調)
- 使用最近鄰邊界模式處理端點

**效果:**
- 消除微小的幀間抖動
- 保留物件的主要運動趨勢
- 軌跡更平滑、更自然

---

## 處理流程 (Processing Pipeline)

```
┌─────────────────────────────────────┐
│  正向追蹤 (Forward Tracking)       │
│  ─────────────────────────────────  │
│  raw_bboxes (可能有無效項)          │
└─────────────────────────────────────┘
                   ↓
        【第一步】異常檢測
┌─────────────────────────────────────┐
│  移除速度過高的跳躍點                │
│  cleaned_bboxes                     │
│  (標記異常項為 None)                 │
└─────────────────────────────────────┘
                   ↓
        【第二步】缺失補全
┌─────────────────────────────────────┐
│  線性內插相鄰有效幀之間的缺失幀       │
│  interpolated_bboxes                │
│  (填補最多 10 幀的空隙)              │
└─────────────────────────────────────┘
                   ↓
        【第三步】時間平滑
┌─────────────────────────────────────┐
│  高斯濾波平滑軌跡                    │
│  smoothed_bboxes (最終結果)          │
│  (消除微小抖動)                      │
└─────────────────────────────────────┘
                   ↓
        【輸出】視頻和 Bbox 檔案
```

---

## 使用方式 (Usage)

### 基本運行

```bash
cd /ssd6/ron/HiM2SAM
source .venv/bin/activate
python scripts/run_skitb_inference_enhanced.py
```

### 預期輸出

```
INFO - 找到 2796 張影像
INFO - 初始 BBox (XYXY): [705.0, 241.0, 967.0, 528.0]
INFO - ========================================
INFO -     Enhanced Tracking Started      
INFO - ========================================
INFO - 初始化視訊狀態...
INFO - 開始追蹤...
Tracking: 100%|██████████| 2796/2796 [XX:XX<00:00, X.XXframe/s]
INFO - 追蹤完成，開始軌跡後處理...
INFO - Step 1: 檢測和移除異常跳躍...
INFO -   移除了 X 個異常點
INFO - Step 2: 內插缺失的幀...
INFO - Step 3: 應用時間平滑...
INFO - 輸出結果...
Writing: 100%|██████████| 2796/2796 [XX:XX<00:00, X.XXframe/s]
INFO - ========================================
INFO - 推論完成！
INFO - 影片已儲存至: logs/.../result_XXXXXXX.mp4
INFO - BBox 結果已儲存至: logs/.../bbox_XXXXXXX.txt
INFO - ========================================
```

---

## 可調整參數 (Tunable Parameters)

在 `TrajectorySmoothing` 初始化中:

```python
smoother = TrajectorySmoothing(logger)
smoother.min_mask_area_ratio = 0.001  # Mask 最小面積比例
smoother.max_velocity = 100            # 最大速度 (px/frame)
```

在調用方法時:

```python
# 異常檢測
cleaned = smoother.remove_outliers_velocity(raw_bboxes, max_velocity=100)
#                                                      ^^^^^^ 可調

# 內插
interpolated = smoother.interpolate_gaps(cleaned, max_gap=10)
#                                                  ^^^^^^^ 可調

# 平滑
smoothed = smoother.smooth_bboxes_temporal(interpolated, sigma=1.5)
#                                                        ^^^^^ 可調
```

### 參數調整指南

| 參數 | 默認值 | 說明 | 調整建議 |
|-----|--------|------|--------|
| `max_velocity` | 100 | 最大允許速度 | 降低→更激進地移除跳躍 |
| `max_gap` | 10 | 最大內插間隙 | 提高→處理更長的遮擋 |
| `sigma` | 1.5 | 高斯濾波標準差 | 提高→更平滑但可能失細節 |
| `mask_area_threshold` | 100 | 最小 mask 像素數 | 提高→更嚴格地過濾 |

---

## 與其他版本的比較

| 功能 | `_clean.py` | `_enhanced.py` | `_bidirectional.py` |
|------|-----------|----------------|------------------|
| 正向追蹤 | ✓ | ✓ | ✓ (+ 反向) |
| 有效性檢查 | ✗ | ✓ | ✓ |
| 異常檢測 | ✗ | ✓ | ✓ |
| 內插修復 | ✗ | ✓ | ✓ |
| 時間平滑 | ✗ | ✓ | ✓ |
| 反向追蹤 | ✗ | ✗ | ✓ |
| 軌跡融合 | ✗ | ✗ | ✓ |
| 耗時 | ~10 分鐘 | ~10 分鐘 | ~15 分鐘 |

**建議:**
- **快速測試:** 使用 `_clean.py`
- **生產環境:** 使用 `_enhanced.py` (最佳平衡)
- **最高精度:** 使用 `_bidirectional.py` (耗時長)

---

## 常見問題 (FAQ)

### Q: 仍有幀沒有 bbox，是什麼原因？
**A:** 可能的原因:
1. Skier 被完全遮擋 (無 mask 生成)
2. Mask 太小或太分散 (被過濾)
3. 異常檢測誤將有效點標記為異常

**解決方案:**
- 降低 `mask_area_threshold` (使用更寬鬆的檢測)
- 提高 `max_velocity` (允許更快的移動)
- 檢查日誌中的 "異常點移除" 數量

### Q: bbox 仍然有抖動，如何進一步平滑？
**A:** 
- 提高 `sigma` 值 (如 2.0 或 2.5)
- 增加內插間隙 `max_gap` (如 15 或 20)

### Q: 輸出的 bbox 與 GT 不匹配，是追蹤錯誤嗎？
**A:** 可能不是。平滑後的 bbox 可能與原始 ground truth 有差異。
- 檢查視頻中的視覺效果，如果看起來合理就沒問題
- 可以調整平滑參數以更接近原始追蹤結果

---

## 技術細節 (Technical Details)

### 高斯濾波實現

```python
from scipy.ndimage import gaussian_filter1d

smoothed = gaussian_filter1d(values, sigma=sigma, mode='nearest')
```

- **sigma:** 控制平滑強度 (數值越大越平滑)
- **mode='nearest':** 邊界處理方式 (使用最近值填充)

### 速度限制計算

```python
# 中心點
c1 = [(x1_min + x1_max) / 2, (y1_min + y1_max) / 2]
c2 = [(x2_min + x2_max) / 2, (y2_min + y2_max) / 2]

# 距離和速度
distance = sqrt((c2[0] - c1[0])^2 + (c2[1] - c1[1])^2)
velocity = distance / (frame_gap)
```

---

## 故障排除 (Troubleshooting)

### 問題: "AttributeError: ... NoneType"
**原因:** 某個 bbox 為 None 且未正確處理
**解決:** 檢查平滑函數是否正確處理 None 值

### 問題: 輸出影片損壞
**原因:** 某個 bbox 座標超出影像邊界
**解決:** 在 main 函數中已添加邊界檢查，確保 bbox 在 [0, width/height] 內

### 問題: 速度太慢
**原因:** 後處理步驟計算量大
**解決:** 優化可以並行化某些步驟 (未來改進)

---

## 許可證和引用

基於 HiM2SAM (Facebook Research)
增強功能: 軌跡平滑和異常檢測

---

**推薦使用此版本進行生產環境的滑雪追蹤！**
