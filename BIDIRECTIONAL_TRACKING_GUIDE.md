# 雙向追蹤系統文檔 (Bidirectional Tracking System Documentation)

## 概述 (Overview)

此系統實現了**雙向追蹤 + 軌跡融合** (Bidirectional Tracking + Trajectory Fusion) 策略，用於處理滑雪追蹤中的場景切換、模糊和遮擋等挑戰。

## 工作流程 (Workflow)

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Forward Tracking                                   │
│  ────────────────────────────────────────────────────────   │
│  使用 HiM2SAM 從第 0 幀到最後一幀執行正向追蹤                │
│  輸出: Forward Bboxes + Confidence Scores                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Backward Tracking                                  │
│  ────────────────────────────────────────────────────────   │
│  將影片反向，從最後一幀回到第 0 幀執行反向追蹤               │
│  輸出: Backward Bboxes + Confidence Scores                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Trajectory Fusion                                  │
│  ────────────────────────────────────────────────────────   │
│  比較正向和反向結果，根據以下規則融合:                       │
│                                                             │
│  - 高 IoU (>0.5): 加權平均兩個結果                         │
│  - 低 IoU: 選擇信心度更高的結果                             │
│  - 缺失: 標記為缺失，後續步驟處理                           │
│                                                             │
│  輸出: Fused Bboxes + Fusion Method Labels                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Interpolation                                      │
│  ────────────────────────────────────────────────────────   │
│  在相鄰的有效點之間進行線性內插，填補最多 10 幀的缺失 bbox   │
│  輸出: Interpolated Bboxes                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Temporal Smoothing                                 │
│  ────────────────────────────────────────────────────────   │
│  應用高斯濾波進行時間平滑，消除微小跳動                      │
│  輸出: Final Smoothed Bboxes                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
                    Output Video & Bbox File
```

## 關鍵概念 (Key Concepts)

### 1. 融合規則 (Fusion Rules)

| 場景 | 正向結果 | 反向結果 | 融合策略 |
|------|---------|---------|--------|
| **一致** | ✓ (高信心度) | ✓ (高信心度) IoU > 0.5 | 加權平均 |
| **衝突** | ✓ (低信心度) | ✓ (高信心度) IoU < 0.5 | 採用反向結果 |
| **正向失敗** | ✗ (無/低信心度) | ✓ (高信心度) | 採用反向結果 |
| **反向失敗** | ✓ (高信心度) | ✗ (無/低信心度) | 採用正向結果 |
| **雙方失敗** | ✗ | ✗ | 標記為缺失 |

### 2. 信心度評分 (Confidence Scoring)

信心度計算基於 **Mask 的填充比例** (Fill Ratio):

```
confidence = min(1.0, (mask_area / total_area) × 2)
```

- **高信心度** (0.7-1.0): 清晰、大面積的 mask
- **中信心度** (0.4-0.7): 部分遮擋或邊界不清晰
- **低信心度** (0-0.4): 非常小或模糊的 mask

### 3. IoU 閾值

- **融合條件:** IoU > 0.5 (表示正向和反向結果高度一致)
- **內插條件:** 最多跨越 10 幀 (合理的遮擋/模糊持續時間)

## 運行方式 (Usage)

### 基本運行

```bash
cd /ssd6/ron/HiM2SAM
source .venv/bin/activate
CUDA_VISIBLE_DEVICE=2 python scripts/run_bidirectional_tracking.py
```

### 預期輸出

運行完成後，在 `logs/<timestamp>_bidirectional_tracking/` 目錄中會生成:

1. **result_<timestamp>.mp4** - 帶 bbox 和 mask 的輸出影片
2. **bbox_<timestamp>.txt** - 每幀的 bbox 結果 (x,y,w,h 格式)
3. **bidirectional_tracking.log** - 詳細的追蹤日誌

### 典型日誌輸出

```
INFO - 找到 2796 張影像
INFO - 初始 BBox (XYXY): [705.0, 241.0, 967.0, 528.0]
INFO - 最後 BBox (XYXY): [...]
INFO - ========================================
INFO -    Bidirectional Tracking Started    
INFO - ========================================
INFO - Running forward tracking pass...
Frames:   0%|          | 0/2796 [00:00<?, ?frame/s]
...
INFO - Forward pass completed.
INFO - Running backward tracking pass...
Frames:   0%|          | 0/2796 [00:00<?, ?frame/s]
...
INFO - Backward pass completed.
INFO - Fusing trajectories...
INFO - Trajectory fusion completed.
INFO - Interpolating missing boxes...
INFO - Interpolation completed.
INFO - Smoothing trajectory...
INFO - Smoothing completed.
INFO - Writing output video and bbox file...
...
INFO - 推論完成！
```

## 可調整參數 (Tunable Parameters)

在 `TrajectoryFusion` 類中:

```python
fusion = TrajectoryFusion(
    logger,
    iou_threshold=0.5,           # IoU 融合閾值 (越高越保守)
    confidence_threshold=0.1     # 信心度最低閾值
)
```

在內插函數中:

```python
interpolated_bboxes = fusion.interpolate_missing_boxes(
    fused_data["bboxes"], 
    fused_data["confidences"], 
    max_gap=10  # 最大內插幀距 (調高可處理更長的遮擋)
)
```

在平滑函數中:

```python
smoothed_bboxes = fusion.smooth_trajectory(
    interpolated_bboxes, 
    sigma=1.5  # 高斯濾波標準差 (越高越平滑但可能丟失細節)
)
```

## 性能指標 (Performance Metrics)

### 預計耗時

- **正向追蹤:** ~6 分鐘 (2796 幀)
- **反向追蹤:** ~6 分鐘 
- **融合 + 內插 + 平滑:** ~1 分鐘
- **輸出影片:** ~1 分鐘
- **總耗時:** ~14-16 分鐘

### 記憶體使用

- **峰值 GPU 記憶體:** ~20 GB (使用 bfloat16 精度)
- **VRAM 需求:** RTX A100 或同級別

## 故障排除 (Troubleshooting)

### 問題 1: 某些幀仍然沒有 bbox

**原因:** 正向和反向追蹤都失敗，且無法通過內插修復。

**解決方案:**
1. 增加 `max_gap` 參數（允許更長的內插距離）
2. 降低 `iou_threshold`（更容易融合結果）
3. 檢查初始化 bbox 是否正確

### 問題 2: 輸出影片中的 bbox 抖動

**原因:** 軌跡在幀之間跳躍。

**解決方案:**
1. 增加 `sigma` 參數（更強的高斯平滑）
2. 檢查是否有可靠的初始化和最終 bbox

### 問題 3: 記憶體不足 (OOM)

**原因:** 影片幀過多或 GPU 記憶體不足。

**解決方案:**
1. 分段處理影片 (暫未支援)
2. 使用 float32 而不是 bfloat16 (速度會變慢)

## 進階技巧 (Advanced Tips)

### 1. 多幀初始化

如果第一幀或最後一幀的 bbox 不夠清晰，可以修改初始化邏輯以使用多個初始化框。

### 2. 場景感知融合

可以在融合時加入場景變化檢測，對不同相機的幀進行不同的融合權重。

### 3. 特定幀的手動修正

如果某幀的結果仍然不正確，可以在最後階段手動編輯 bbox txt 檔案。

## 理論背景 (Theoretical Background)

該系統基於以下假設:

1. **互補性 (Complementarity):** 正向和反向追蹤的錯誤不會發生在同一幀。
2. **記憶體有效性 (Memory Validity):** HiM2SAM 的長期記憶庫能有效儲存和檢索目標特徵。
3. **運動連續性 (Motion Continuity):** 目標的運動軌跡在時間上是連續且平滑的。

在滑雪追蹤的場景中，這些假設通常成立，因為:

- 即使正向追蹤在場景切換時失敗，反向追蹤可以從未來的清晰幀進行回溯。
- HiM2SAM 的層次化記憶設計特別適合處理視覺上的外觀變化。
- 滑雪者的運動軌跡通常遵循物理規律，異常跳躍是罕見的。

## 參考文獻 (References)

- **HiM2SAM:** Ye et al., "HiM2SAM: Hierarchical Multi-Memory Networks for Video Object Tracking" (2024)
- **雙向追蹤:** 標準的離線追蹤技術，廣泛應用於電影、體育分析等領域
- **軌跡融合:** 常見於多模型融合和重新識別系統

## 聯繫與支援 (Support)

如有問題，請檢查以下文件:

1. `logs/<timestamp>_bidirectional_tracking/bidirectional_tracking.log` - 詳細日誌
2. `scripts/run_bidirectional_tracking.py` - 原始代碼
3. SAM2 官方文檔 (https://github.com/facebookresearch/sam2)

---

**最後更新:** 2025-12-08  
**版本:** 1.0
