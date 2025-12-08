# Enhanced HiM2SAM Inference with Trajectory Smoothing
# CUDA_VISIBLE_DEVICE=2 python scripts/run_skitb_inference_enhanced.py
import cv2
import gc
import numpy as np
import os
import torch
import sys
import tqdm
import logging
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

# 取得專案根目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root in sys.path:
    sys.path.remove(project_root)

from sam2.build_sam import build_sam2_video_predictor

# Configure logging
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "inference.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    
    return logger

def load_boxes(gt_path):
    """讀取 boxes.txt"""
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"找不到 bbox 檔案: {gt_path}")
        
    boxes = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()
            
            try:
                x, y, w, h = map(float, parts)
                boxes.append([x, y, w, h])
            except ValueError:
                pass
    return boxes

def load_cameras(cam_path):
    """讀取 cameras.txt"""
    if not os.path.exists(cam_path):
        raise FileNotFoundError(f"找不到 cameras 檔案: {cam_path}")
    
    cameras = []
    with open(cam_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cameras.append(line)
    return cameras

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x+w, y+h]

class SimpleKalmanFilter:
    """簡單的卡爾曼濾波器用於 bbox 平滑"""
    
    def __init__(self, process_variance=0.04, measurement_variance=0.04):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state = None
        self.error_estimate = 1.0
    
    def filter_value(self, measurement):
        """平滑單個數值"""
        if self.state is None:
            self.state = measurement
            return measurement
        
        # 預測
        prediction = self.state
        
        # 更新誤差估計
        self.error_estimate += self.process_variance
        
        # 卡爾曼增益
        kalman_gain = self.error_estimate / (self.error_estimate + self.measurement_variance)
        
        # 更新狀態
        self.state = prediction + kalman_gain * (measurement - prediction)
        
        # 更新誤差估計
        self.error_estimate = (1 - kalman_gain) * self.error_estimate
        
        return self.state
    
    def reset(self):
        self.state = None
        self.error_estimate = 1.0

class TrajectorySmoothing:
    """軌跡平滑引擎"""
    
    def __init__(self, logger):
        self.logger = logger
        self.min_mask_area_ratio = 0.001  # 最小 mask 面積比例 (總像素 * 0.1%)
        self.max_velocity = 100  # 最大速度 (像素/幀)
    
    def detect_valid_mask(self, mask, mask_area_threshold=100):
        """
        檢測 mask 是否有效
        
        Returns:
            is_valid: bool
            bbox: [x1, y1, x2, y2] or None
            area_ratio: float
        """
        
        if mask is None or mask.sum() == 0:
            return False, None, 0.0
        
        mask_area = int(mask.sum())
        total_area = mask.size
        area_ratio = mask_area / total_area
        
        # 檢查最小面積閾值
        if mask_area < mask_area_threshold:
            return False, None, area_ratio
        
        # 提取 bbox
        non_zero_indices = np.argwhere(mask)
        if len(non_zero_indices) == 0:
            return False, None, area_ratio
        
        y_min, x_min = non_zero_indices.min(axis=0).tolist()
        y_max, x_max = non_zero_indices.max(axis=0).tolist()
        
        # 檢查 bbox 大小
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        if bbox_w < 10 or bbox_h < 10:  # 太小的 bbox 認為無效
            return False, None, area_ratio
        
        return True, [x_min, y_min, x_max, y_max], area_ratio
    
    def smooth_bboxes_temporal(self, bboxes, window_size=5, sigma=1.0):
        """
        對軌跡進行時間平滑 (高斯濾波)
        
        Args:
            bboxes: List of [x1, y1, x2, y2] or None
            window_size: 高斯窗口大小
            sigma: 高斯標準差
        
        Returns:
            smoothed_bboxes: List of smoothed bboxes or None
        """
        
        n = len(bboxes)
        valid_indices = [i for i, bbox in enumerate(bboxes) if bbox is not None]
        
        if len(valid_indices) < 2:
            return bboxes
        
        smoothed_bboxes = [bbox for bbox in bboxes]
        
        # 對每個座標分別進行平滑
        for coord_idx in range(4):  # x1, y1, x2, y2
            coords = []
            for bbox in bboxes:
                if bbox is not None:
                    coords.append(bbox[coord_idx])
                else:
                    # 用鄰近有效值填充
                    if len(coords) > 0:
                        coords.append(coords[-1])
                    else:
                        coords.append(0)
            
            # 應用高斯濾波
            smoothed_coords = gaussian_filter1d(coords, sigma=sigma, mode='nearest')
            
            # 將平滑後的值寫回
            coord_idx_local = 0
            for i, bbox in enumerate(bboxes):
                if bbox is not None:
                    if coord_idx == 0:
                        smoothed_bboxes[i] = list(bbox)
                    smoothed_bboxes[i][coord_idx] = float(smoothed_coords[i])
                coord_idx_local += 1
        
        return smoothed_bboxes
    
    def remove_outliers_velocity(self, bboxes, max_velocity=100):
        """
        根據速度限制移除異常的 bbox (跳躍檢測)
        
        Args:
            bboxes: List of [x1, y1, x2, y2] or None
            max_velocity: 最大允許速度 (像素/幀)
        
        Returns:
            cleaned_bboxes: List with outliers marked as None
        """
        
        cleaned = [bbox for bbox in bboxes]
        
        # 計算相鄰幀的中心點距離
        valid_indices = [i for i, bbox in enumerate(cleaned) if bbox is not None]
        
        if len(valid_indices) < 2:
            return cleaned
        
        for i in range(len(valid_indices) - 1):
            idx1 = valid_indices[i]
            idx2 = valid_indices[i + 1]
            
            if idx2 - idx1 > 5:  # 間隙太大，跳過
                continue
            
            bbox1 = cleaned[idx1]
            bbox2 = cleaned[idx2]
            
            # 重新驗證 bbox 存在 (防止修改後的列表產生 None)
            if bbox1 is None or bbox2 is None:
                continue
            
            # 計算中心點
            c1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
            c2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
            
            # 計算距離和速度
            distance = np.linalg.norm(c2 - c1)
            velocity = distance / (idx2 - idx1)
            
            # 如果速度過高，標記為異常
            if velocity > max_velocity:
                self.logger.warning(f"Outlier detected at frame {idx2}: velocity={velocity:.2f} px/frame (max={max_velocity})")
                cleaned[idx2] = None
        
        return cleaned
    
    def interpolate_gaps(self, bboxes, max_gap=10):
        """
        在缺失的幀之間進行線性內插
        
        Args:
            bboxes: List of [x1, y1, x2, y2] or None
            max_gap: 最大內插間隙
        
        Returns:
            interpolated_bboxes: List with gaps filled
        """
        
        interpolated = [bbox for bbox in bboxes]
        valid_indices = [i for i, bbox in enumerate(interpolated) if bbox is not None]
        
        if len(valid_indices) < 2:
            return interpolated
        
        for i in range(len(valid_indices) - 1):
            idx1 = valid_indices[i]
            idx2 = valid_indices[i + 1]
            gap = idx2 - idx1
            
            if 1 < gap <= max_gap:
                bbox1 = interpolated[idx1]
                bbox2 = interpolated[idx2]
                
                for k in range(1, gap):
                    alpha = k / gap
                    interpolated_bbox = [
                        bbox1[j] * (1 - alpha) + bbox2[j] * alpha
                        for j in range(4)
                    ]
                    interpolated[idx1 + k] = interpolated_bbox
        
        return interpolated

# 設定路徑
metadata_dir = os.path.join(project_root, "test-data/AL0098/MC")
cameras_path = os.path.join(metadata_dir, "cameras.txt")
boxes_path = os.path.join(metadata_dir, "boxes.txt")
video_frames_dir = "/ssd6/ron/SkiTB/AL/AL0098/frames"

# 輸出設定
exp_name = "skitb_inference_enhanced"
timestamp = datetime.now().strftime("%m%d%H%M%S")
log_parent_dir = os.path.join(project_root, "logs")
output_dir = os.path.join(log_parent_dir, f"{timestamp}_{exp_name}")
os.makedirs(output_dir, exist_ok=True)

logger = setup_logging(output_dir)

output_video_path = os.path.join(output_dir, f"result_{timestamp}.mp4")
bbox_output_path = os.path.join(output_dir, f"bbox_{timestamp}.txt")

# 模型設定
model_name = "large"
checkpoint_path = os.path.join(project_root, "checkpoints", "sam2.1_hiera_large.pt")
model_cfg = "configs/him2sam/lasot/sam2.1_hiera_l.yaml"

color = (0, 255, 0)  # Green

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

def main():
    # 1. 準備影像列表
    if not os.path.exists(video_frames_dir):
        logger.error(f"錯誤: 找不到影像目錄 {video_frames_dir}")
        return

    frame_names = sorted([p for p in os.listdir(video_frames_dir) 
                         if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]])
    if not frame_names:
        logger.error("目錄中沒有找到影像檔")
        return
        
    num_frames = len(frame_names)
    logger.info(f"找到 {num_frames} 張影像")
    
    first_img_path = os.path.join(video_frames_dir, frame_names[0])
    first_img = cv2.imread(first_img_path)
    height, width = first_img.shape[:2]

    # 2. 載入 metadata
    try:
        all_boxes = load_boxes(boxes_path)
        all_cameras = load_cameras(cameras_path)
        
        if len(all_boxes) == 0:
            logger.error("Boxes file is empty.")
            return
             
        if len(all_boxes) != num_frames or len(all_cameras) != num_frames:
            logger.warning("Data length mismatch! Using min length...")
            max_safe_idx = min(num_frames, len(all_boxes), len(all_cameras))
        else:
            max_safe_idx = num_frames
            
        init_bbox = xywh_to_xyxy(all_boxes[0])
        logger.info(f"初始 BBox (XYXY): {init_bbox}")
        
    except Exception as e:
        logger.error(f"讀取 metadata 失敗: {e}")
        return

    # 3. 初始化 SAM 2 預測器
    logger.info(f"載入模型: {model_cfg} ...")
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)

    logger.info("=" * 40)
    logger.info("      Enhanced Tracking Started      ")
    logger.info("=" * 40)
    logger.info(f"num_maskmem: {getattr(predictor, 'num_maskmem', 'N/A')}")
    logger.info(f"rvcot_mem_long_len: {getattr(predictor, 'rvcot_mem_long_len', 'N/A')}")
    logger.info("=" * 40)

    # 4. 初始化 Video Writer 和 BBox 檔案
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
    bbox_file = open(bbox_output_path, 'w')
    
    # 初始化軌跡平滑引擎
    smoother = TrajectorySmoothing(logger)
    
    # 暫存所有 bbox (用於後處理)
    raw_bboxes = []

    # 5. 推論迴圈
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pbar = tqdm.tqdm(total=max_safe_idx, desc="Tracking", unit="frame", dynamic_ncols=True)
        
        logger.info("初始化視訊狀態...")
        state = predictor.init_state(video_frames_dir, offload_video_to_cpu=True, 
                                    offload_state_to_cpu=True, async_loading_frames=True)

        # 加入第一幀的提示
        _, _, _ = predictor.add_new_points_or_box(state, box=init_bbox, frame_idx=0, obj_id=1)

        logger.info("開始追蹤...")
        
        prop_gen = predictor.propagate_in_video(state, start_frame_idx=0)
        
        for frame_idx, object_ids, masks in prop_gen:
            
            # 讀取影像
            img_path = os.path.join(video_frames_dir, frame_names[frame_idx])
            img = cv2.imread(img_path)
            
            if img is None:
                logger.warning(f"警告: 無法讀取影像 {img_path}")
                raw_bboxes.append(None)
                break

            # 處理 Mask
            bbox = None
            if object_ids is not None and len(object_ids) > 0 and masks is not None and len(masks) > 0:
                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy()
                    mask = mask > 0.0
                    
                    # 檢測有效 mask
                    is_valid, detected_bbox, area_ratio = smoother.detect_valid_mask(mask, mask_area_threshold=100)
                    
                    if is_valid:
                        bbox = detected_bbox
                        break  # 使用第一個有效 bbox
            
            raw_bboxes.append(bbox)
            pbar.update(1)

        pbar.close()

    logger.info("追蹤完成，開始軌跡後處理...")
    
    # 6. 軌跡後處理
    # Step 1: 移除速度異常的 bbox (跳躍檢測)
    logger.info("Step 1: 檢測和移除異常跳躍...")
    cleaned_bboxes = smoother.remove_outliers_velocity(raw_bboxes, max_velocity=100)
    outlier_count = sum(1 for i in range(len(raw_bboxes)) 
                        if raw_bboxes[i] is not None and cleaned_bboxes[i] is None)
    logger.info(f"  移除了 {outlier_count} 個異常點")
    
    # Step 2: 內插缺失的 bbox
    logger.info("Step 2: 內插缺失的幀...")
    interpolated_bboxes = smoother.interpolate_gaps(cleaned_bboxes, max_gap=10)
    
    # Step 3: 時間平滑
    logger.info("Step 3: 應用時間平滑...")
    smoothed_bboxes = smoother.smooth_bboxes_temporal(interpolated_bboxes, window_size=5, sigma=1.5)
    
    # 7. 輸出結果
    logger.info("輸出結果...")
    pbar = tqdm.tqdm(total=max_safe_idx, desc="Writing", unit="frame", dynamic_ncols=True)
    
    for i in range(max_safe_idx):
        img_path = os.path.join(video_frames_dir, frame_names[i])
        img = cv2.imread(img_path)
        
        if img is None:
            pbar.update(1)
            continue
        
        bbox = smoothed_bboxes[i] if i < len(smoothed_bboxes) else None
        
        if bbox is not None:
            x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            
            # 確保 bbox 在影像範圍內
            x_min = max(0, min(x_min, width))
            y_min = max(0, min(y_min, height))
            x_max = max(0, min(x_max, width))
            y_max = max(0, min(y_max, height))
            
            bbox_file.write(f"{x_min},{y_min},{bbox_w},{bbox_h}\n")
            
            # 繪製 BBox
            if x_max > x_min and y_max > y_min:
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)
                
                # 繪製 mask 覆蓋
                mask_overlay = np.zeros((height, width, 3), np.uint8)
                mask_overlay[y_min:y_max, x_min:x_max] = color
                img = cv2.addWeighted(img, 1, mask_overlay, 0.2, 0)
        else:
            bbox_file.write("0,0,0,0\n")

        out.write(img)
        pbar.update(1)

    pbar.close()

    # 8. 清理
    out.release()
    bbox_file.close()
    
    logger.info("=" * 40)
    logger.info("推論完成！")
    logger.info(f"影片已儲存至: {output_video_path}")
    logger.info(f"BBox 結果已儲存至: {bbox_output_path}")
    logger.info("=" * 40)
    
    del predictor
    del state
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
