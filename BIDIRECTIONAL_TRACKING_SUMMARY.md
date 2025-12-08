# Quick summary of the Bidirectional Tracking Implementation
# 雙向追蹤系統實現總結

## 核心模塊 (Core Modules)

### 1. TrajectoryFusion 類
- **功能:** 融合正向和反向追蹤結果
- **方法:**
  - `fuse_trajectories()`: 逐幀融合決策
  - `interpolate_missing_boxes()`: 線性內插缺失 bbox
  - `smooth_trajectory()`: 時間域高斯平滑
  - `_weighted_average_bbox()`: 加權平均兩個 bbox

### 2. 融合策略
- **高 IoU (>0.5):** 加權平均（信心度更高的權重更大）
- **低 IoU (<0.5):** 選擇信心度更高的結果
- **單側失敗:** 採用有效的一側
- **雙側失敗:** 標記為缺失，後續由內插處理

### 3. 信心度計算
```
confidence = min(1.0, (mask_area / total_area) × 2)
```
基於 mask 的填充比例，範圍 [0, 1]

### 4. 內插和平滑
- **內插:** 線性插值，最多跨越 10 幀
- **平滑:** 高斯濾波，sigma=1.5

## 使用方式 (Usage)

```bash
cd /ssd6/ron/HiM2SAM
source .venv/bin/activate
python scripts/run_bidirectional_tracking.py
```

## 預期時間和資源 (Expected Time & Resources)

- **正向追蹤:** ~6 分鐘
- **反向追蹤:** ~6 分鐘
- **融合+內插+平滑:** ~2 分鐘
- **輸出:** ~1 分鐘
- **總耗時:** ~15 分鐘

**GPU 要求:** RTX A100 或同級別 (20GB VRAM)

## 主要優勢 (Key Advantages)

✅ 利用 HiM2SAM 的長期記憶庫處理場景變化
✅ 正向和反向結果互補，提高魯棒性
✅ 自動檢測並修正追蹤錯誤
✅ 時間平滑確保軌跡連續性
✅ 無需額外手工標註，完全自動化

## 下一步改進 (Future Improvements)

1. 加入場景變化檢測，不同相機區間採用不同策略
2. 實現分段處理以支援更長的影片
3. 加入用戶互動界面，手動修正特定幀
4. 優化記憶體使用，支援 float32 精度
5. 並行化正向和反向追蹤以縮短耗時

---

完整實現在: `/ssd6/ron/HiM2SAM/scripts/run_bidirectional_tracking.py`
詳細指南在: `/ssd6/ron/HiM2SAM/BIDIRECTIONAL_TRACKING_GUIDE.md`
