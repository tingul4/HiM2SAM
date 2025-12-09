# Bidirectional Tracking with Trajectory Fusion for SkiTB
# 正向追蹤 + 反向追蹤 + 軌跡融合
import cv2
import gc
import numpy as np
import os
import torch
import sys
import tqdm
import logging
from datetime import datetime
from scipy.interpolate import interp1d
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
    log_file_path = os.path.join(log_dir, "bidirectional_tracking.log")

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

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2-x1, y2-y1]

def compute_iou(box1, box2):
    """計算兩個 bbox 的 IoU (xyxy 格式)"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def compute_trajectory_smoothness(trajectory, window_size=5):
    """
    計算軌跡的平滑度 (基於二階導數)
    返回每個點的 smoothness score (0-1, 越平滑分數越高)
    """
    n = len(trajectory)
    smoothness = np.ones(n)
    
    if n < window_size:
        return smoothness
    
    for i in range(window_size // 2, n - window_size // 2):
        # 提取 center, x, y 座標
        centers = np.array([traj[2:4] for traj in trajectory[i-window_size//2:i+window_size//2+1]])
        
        # 計算一階導數 (速度)
        velocities = np.diff(centers, axis=0)
        
        # 計算二階導數 (加速度)
        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0)
            # 加速度越小，軌跡越平滑
            acc_magnitude = np.mean(np.linalg.norm(accelerations, axis=1))
            # 正規化為 0-1 (假設 max acceleration 為 50 pixels)
            smoothness[i] = max(0, 1 - (acc_magnitude / 50.0))
    
    return smoothness

class TrajectoryFusion:
    """軌跡融合引擎"""
    
    def __init__(self, logger, iou_threshold=0.5, confidence_threshold=0.1):
        self.logger = logger
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.min_mask_area = 100  # 最小 mask 面積閾值
        self.min_bbox_size = 10   # 最小 bbox 尺寸
        self.max_velocity = 100   # 最大速度 (像素/幀)
        self.camera_switch_window = 3  # 相機切換後的觀察窗口（幀數）
    
    def detect_valid_mask(self, mask, bbox):
        """
        檢測 mask 和 bbox 是否有效
        
        Returns:
            is_valid: bool
        """
        if mask is None:
            return False
        
        # 檢查 mask 面積
        mask_area = int(mask.sum())
        if mask_area < self.min_mask_area:
            return False
        
        # 檢查 bbox 尺寸
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            
            if bbox_w < self.min_bbox_size or bbox_h < self.min_bbox_size:
                return False
        
        return True
    
    def remove_outliers_velocity(self, bboxes, max_velocity=None):
        """
        根據速度限制移除異常的 bbox (跳躍檢測)
        
        Args:
            bboxes: List of [x1, y1, x2, y2] or None
            max_velocity: 最大允許速度 (像素/幀)
        
        Returns:
            cleaned_bboxes: List with outliers marked as None
        """
        if max_velocity is None:
            max_velocity = self.max_velocity
        
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
            
            # 重新驗證 bbox 存在
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
    
    def adjust_confidence_near_switches(self, confidences, camera_switches):
        """
        調整相機切換附近的信心度
        在切換點附近降低信心度，因為 HiM2SAM 的長期記憶可能需要時間重新適應
        
        Args:
            confidences: List of confidence values
            camera_switches: List of frame indices where camera switches occur
        
        Returns:
            adjusted_confidences: List with adjusted values
        """
        if not camera_switches:
            return confidences
        
        adjusted = list(confidences)
        
        for switch_frame in camera_switches:
            # 在切換點前後的窗口內降低信心度
            start_idx = max(0, switch_frame - self.camera_switch_window)
            end_idx = min(len(adjusted), switch_frame + self.camera_switch_window + 1)
            
            for i in range(start_idx, end_idx):
                # 距離切換點越近，信心度降低越多
                distance = abs(i - switch_frame)
                penalty = max(0.3, 1.0 - (distance / self.camera_switch_window) * 0.5)
                adjusted[i] = adjusted[i] * penalty
        
        return adjusted
    
    def fuse_trajectories(self, forward_data, backward_data, frame_count):
        """
        融合正向和反向追蹤結果
        
        Args:
            forward_data: {"bboxes": [...], "confidences": [...]}
            backward_data: {"bboxes": [...], "confidences": [...]}
            frame_count: 總幀數
        
        Returns:
            fused_data: {"bboxes": [...], "confidences": [...], "fusion_method": [...]}
        """
        
        forward_bboxes = forward_data.get("bboxes", [])
        forward_conf = forward_data.get("confidences", [])
        backward_bboxes = backward_data.get("bboxes", [])
        backward_conf = backward_data.get("confidences", [])
        
        # 反向數據需要翻轉回原序列
        backward_bboxes = backward_bboxes[::-1]
        backward_conf = backward_conf[::-1]
        
        fused_bboxes = []
        fused_conf = []
        fusion_method = []
        
        for i in range(frame_count):
            fwd_bbox = forward_bboxes[i] if i < len(forward_bboxes) else None
            fwd_conf = forward_conf[i] if i < len(forward_conf) else 0.0
            
            bwd_bbox = backward_bboxes[i] if i < len(backward_bboxes) else None
            bwd_conf = backward_conf[i] if i < len(backward_conf) else 0.0
            
            # --- 情況 1: 雙方都有有效結果 ---
            if fwd_bbox is not None and bwd_bbox is not None:
                iou = compute_iou(fwd_bbox, bwd_bbox)
                
                if iou > self.iou_threshold:
                    # 高度一致，融合兩個結果
                    fused_bbox = self._weighted_average_bbox(
                        fwd_bbox, fwd_conf,
                        bwd_bbox, bwd_conf
                    )
                    fused_conf_val = (fwd_conf + bwd_conf) / 2.0
                    fusion_method.append("fused")
                else:
                    # 不一致，選擇信心度更高的
                    if fwd_conf >= bwd_conf:
                        fused_bbox = fwd_bbox
                        fused_conf_val = fwd_conf
                        fusion_method.append("forward_preferred")
                    else:
                        fused_bbox = bwd_bbox
                        fused_conf_val = bwd_conf
                        fusion_method.append("backward_preferred")
            
            # --- 情況 2: 只有一方有結果 ---
            elif fwd_bbox is not None:
                fused_bbox = fwd_bbox
                fused_conf_val = fwd_conf
                fusion_method.append("forward_only")
            elif bwd_bbox is not None:
                fused_bbox = bwd_bbox
                fused_conf_val = bwd_conf
                fusion_method.append("backward_only")
            
            # --- 情況 3: 雙方都沒有結果 ---
            else:
                fused_bbox = None
                fused_conf_val = 0.0
                fusion_method.append("missing")
            
            fused_bboxes.append(fused_bbox)
            fused_conf.append(fused_conf_val)
        
        return {
            "bboxes": fused_bboxes,
            "confidences": fused_conf,
            "fusion_method": fusion_method
        }
    
    def _weighted_average_bbox(self, bbox1, conf1, bbox2, conf2):
        """加權平均兩個 bbox"""
        total_conf = conf1 + conf2
        if total_conf == 0:
            return bbox1
        
        w1 = conf1 / total_conf
        w2 = conf2 / total_conf
        
        averaged = [
            w1 * bbox1[0] + w2 * bbox2[0],
            w1 * bbox1[1] + w2 * bbox2[1],
            w1 * bbox1[2] + w2 * bbox2[2],
            w1 * bbox1[3] + w2 * bbox2[3],
        ]
        return averaged
    
    def interpolate_missing_boxes(self, bboxes, confidences, max_gap=10):
        """
        內插缺失的 bbox
        只在相鄰的可靠點之間進行線性內插
        """
        
        interpolated_bboxes = [bbox for bbox in bboxes]
        
        # 找出所有有效點的索引
        valid_indices = [i for i, bbox in enumerate(bboxes) if bbox is not None]
        
        if len(valid_indices) < 2:
            return interpolated_bboxes
        
        # 在相鄰的有效點之間進行內插
        for j in range(len(valid_indices) - 1):
            idx1 = valid_indices[j]
            idx2 = valid_indices[j + 1]
            gap = idx2 - idx1
            
            # 只在合理的間隙內進行內插
            if gap <= max_gap and gap > 1:
                bbox1 = bboxes[idx1]
                bbox2 = bboxes[idx2]
                
                # 安全檢查（雖然理論上不需要，但增加鲁棒性）
                if bbox1 is None or bbox2 is None:
                    continue
                
                # 線性內插
                for k in range(1, gap):
                    alpha = k / gap
                    interpolated_bbox = [
                        bbox1[0] * (1 - alpha) + bbox2[0] * alpha,
                        bbox1[1] * (1 - alpha) + bbox2[1] * alpha,
                        bbox1[2] * (1 - alpha) + bbox2[2] * alpha,
                        bbox1[3] * (1 - alpha) + bbox2[3] * alpha,
                    ]
                    interpolated_bboxes[idx1 + k] = interpolated_bbox
        
        return interpolated_bboxes
    
    def smooth_trajectory(self, bboxes, sigma=1.5):
        """
        對軌跡進行時間平滑 (高斯濾波)
        """
        
        smoothed_bboxes = []
        valid_bboxes = np.array([bbox for bbox in bboxes if bbox is not None])
        
        if len(valid_bboxes) == 0:
            return bboxes
        
        # 對每個座標進行獨立的高斯濾波
        for dim in range(4):
            # 處理缺失值：用鄰近的有效值填充
            values = []
            for bbox in bboxes:
                if bbox is not None:
                    values.append(bbox[dim])
                else:
                    if len(values) > 0:
                        values.append(values[-1])  # 用上一個有效值填充
                    else:
                        values.append(0)
            
            # 應用高斯濾波
            smoothed_values = gaussian_filter1d(values, sigma=sigma, mode='nearest')
            
            # 儲存平滑後的值
            if dim == 0:
                smoothed_bboxes = [[smoothed_values[i]] for i in range(len(smoothed_values))]
            else:
                for i in range(len(smoothed_bboxes)):
                    smoothed_bboxes[i].append(smoothed_values[i])
        
        return smoothed_bboxes

# 設定路徑
metadata_dir = os.path.join(project_root, "test-data/AL0098/MC")
cameras_path = os.path.join(metadata_dir, "cameras.txt")
boxes_path = os.path.join(metadata_dir, "boxes.txt")
video_frames_dir = "/ssd6/ron/SkiTB/AL/AL0098/frames"

# 輸出設定
exp_name = "bidirectional_tracking"
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

def detect_camera_switches(cameras):
    """
    檢測相機切換的幀索引
    
    Returns:
        List of frame indices where camera switches occur
    """
    switches = []
    for i in range(1, len(cameras)):
        if cameras[i] != cameras[i-1]:
            switches.append(i)
    return switches

def run_tracking_pass(predictor, state, frame_names, video_frames_dir, max_safe_idx, 
                      init_bbox, direction="forward", logger=None, fusion_engine=None, cameras=None):
    """
    執行一次追蹤（正向或反向）
    
    Args:
        fusion_engine: TrajectoryFusion instance for mask validation
        cameras: List of camera names for each frame (for switch detection)
    
    Returns:
        {"bboxes": [...], "confidences": [...], "camera_switches": [...]}
    """
    
    if direction == "forward":
        frame_sequence = range(max_safe_idx)
        start_frame = 0
    else:  # backward
        frame_sequence = range(max_safe_idx - 1, -1, -1)
        start_frame = max_safe_idx - 1
    
    # 檢測相機切換
    camera_switches = []
    if cameras is not None:
        camera_switches = detect_camera_switches(cameras)
        logger.info(f"Detected {len(camera_switches)} camera switches at frames: {camera_switches[:10]}{'...' if len(camera_switches) > 10 else ''}")
    
    logger.info(f"Running {direction} tracking pass...")
    
    bboxes = [None] * max_safe_idx
    confidences = [0.0] * max_safe_idx
    
    # 初始化狀態
    predictor.reset_state(state)
    state = predictor.init_state(video_frames_dir, offload_video_to_cpu=True, 
                                 offload_state_to_cpu=True, async_loading_frames=True)
    
    # 加入初始 bbox（對於反向追蹤，init_bbox 應該是最後一幀的 bbox）
    _, _, _ = predictor.add_new_points_or_box(state, box=init_bbox, frame_idx=start_frame, obj_id=1)
    
    pbar = tqdm.tqdm(total=max_safe_idx, desc=f"{direction.capitalize()} Pass", 
                    unit="frame", dynamic_ncols=True)
    
    prop_gen = predictor.propagate_in_video(
        state, 
        start_frame_idx=start_frame,
        reverse=(direction == "backward")
    )
    
    for frame_idx, object_ids, masks in prop_gen:
        if masks is not None and len(masks) > 0:
            mask = masks[0][0].cpu().numpy() > 0.0
            non_zero_indices = np.argwhere(mask)
            
            # 獲取模型輸出的 object score (logits)
            # 這是 HiM2SAM/SAM2 的內部信心度，比 mask 面積更準確
            model_conf = 0.5 # 預設值
            
            # 嘗試從 state 中獲取 logits
            try:
                obj_idx = state["obj_id_to_idx"].get(1) # 假設 obj_id=1
                if obj_idx is not None and "frame_score_for_mem" in state["output_dict"]:
                    frame_scores = state["output_dict"]["frame_score_for_mem"].get(frame_idx)
                    if frame_scores and "object_score_logits" in frame_scores:
                        logits = frame_scores["object_score_logits"]
                        # logits 可能是 array 或 scalar
                        if np.ndim(logits) > 0 and len(logits) > obj_idx:
                            logit = logits[obj_idx]
                        else:
                            logit = logits
                        
                        # Sigmoid 轉換為 0-1
                        model_conf = 1 / (1 + np.exp(-float(logit)))
                    elif frame_idx in state["consolidated_frame_inds"]["cond_frame_outputs"]:
                        # 如果是 conditioning frame (初始幀)，信心度為 1.0
                        model_conf = 1.0
            except Exception as e:
                # logger.warning(f"Error getting confidence: {e}")
                pass

            # 如果模型信心度太低，視為無效 (解決無 skier 場景出現框框的問題)
            # 閾值 0.0 對應 logit 0.0 (sigmoid(0) = 0.5)，可以根據需要調整
            # 這裡設為 0.4 (logit ~ -0.4) 以過濾掉非常不確定的預測
            if model_conf < 0.4: 
                 bboxes[frame_idx] = None
                 confidences[frame_idx] = 0.0
                 continue

            if len(non_zero_indices) > 0:
                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                
                bbox = [x_min, y_min, x_max, y_max]
                
                # 檢測 mask 和 bbox 的有效性
                is_valid = True
                if fusion_engine is not None:
                    is_valid = fusion_engine.detect_valid_mask(mask, bbox)
                
                if is_valid:
                    bboxes[frame_idx] = bbox
                    confidences[frame_idx] = model_conf
                else:
                    # 無效的檢測，標記為 None
                    bboxes[frame_idx] = None
                    confidences[frame_idx] = 0.0
        
        pbar.update(1)
    
    pbar.close()
    
    return {
        "bboxes": bboxes,
        "confidences": confidences,
        "camera_switches": camera_switches
    }

def main():
    # 準備影像列表
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
    
    # 嘗試解析起始幀索引
    try:
        start_frame_idx = int(os.path.splitext(frame_names[0])[0])
        logger.info(f"起始幀索引: {start_frame_idx}")
    except ValueError:
        start_frame_idx = 0
        logger.warning("無法解析起始幀索引，預設為 0")
    
    first_img_path = os.path.join(video_frames_dir, frame_names[0])
    first_img = cv2.imread(first_img_path)
    height, width = first_img.shape[:2]

    # 載入 metadata
    try:
        all_boxes = load_boxes(boxes_path)
        all_cameras = load_cameras(cameras_path)
        
        if len(all_boxes) == 0:
            logger.error("Boxes file is empty.")
            return
        
        # 處理數據長度不匹配或偏移問題
        # 如果 boxes 數量遠大於 frames 數量，且起始幀索引 > 0，嘗試進行切片
        if len(all_boxes) > num_frames and start_frame_idx > 0:
            if start_frame_idx + num_frames <= len(all_boxes):
                logger.info(f"檢測到數據長度差異，嘗試根據起始幀 {start_frame_idx} 進行切片...")
                all_boxes = all_boxes[start_frame_idx : start_frame_idx + num_frames]
                if len(all_cameras) >= start_frame_idx + num_frames:
                    all_cameras = all_cameras[start_frame_idx : start_frame_idx + num_frames]
            else:
                logger.warning(f"起始幀 {start_frame_idx} 超出數據範圍，使用預設對齊")

        if len(all_boxes) != num_frames or len(all_cameras) != num_frames:
            logger.warning(f"Data length mismatch! Using min length...")
            max_safe_idx = min(num_frames, len(all_boxes), len(all_cameras))
        else:
            max_safe_idx = num_frames
            
        init_bbox = xywh_to_xyxy(all_boxes[0])
        final_bbox = xywh_to_xyxy(all_boxes[-1])
        logger.info(f"初始 BBox (XYXY): {init_bbox}")
        logger.info(f"最後 BBox (XYXY): {final_bbox}")
        
    except Exception as e:
        logger.error(f"讀取 metadata 失敗: {e}")
        return

    # 初始化 SAM 2 預測器
    logger.info(f"載入模型: {model_cfg} ...")
    # 增加 rvcot_mem_long_len 以強化長期記憶，解決追蹤不穩定的問題
    predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device, rvcot_mem_long_len=7)

    logger.info("=" * 40)
    logger.info("   Bidirectional Tracking Started    ")
    logger.info("=" * 40)
    logger.info(f"Memory Configuration:")
    logger.info(f"  num_maskmem: {getattr(predictor, 'num_maskmem', 'N/A')}")
    logger.info(f"  rvcot_mode: {getattr(predictor, 'rvcot_mode', 'N/A')}")
    logger.info(f"  rvcot_mem_long_len: {getattr(predictor, 'rvcot_mem_long_len', 'N/A')} (long-term memory frames)")
    logger.info(f"  rvcot_mem_selection: {getattr(predictor, 'rvcot_mem_selection', 'N/A')}")
    logger.info(f"  rvcot_ious_threshold: {getattr(predictor, 'rvcot_ious_threshold', 'N/A')}")
    logger.info("=" * 40)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
    bbox_file = open(bbox_output_path, 'w')

    # 初始化融合引擎（用於 mask 驗證）
    fusion = TrajectoryFusion(logger, iou_threshold=0.5, confidence_threshold=0.1)

    # === 正向追蹤 ===
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_frames_dir, offload_video_to_cpu=True, 
                                    offload_state_to_cpu=True, async_loading_frames=True)
        forward_data = run_tracking_pass(
            predictor, state, frame_names, video_frames_dir, 
            max_safe_idx, init_bbox, direction="forward", logger=logger,
            fusion_engine=fusion, cameras=all_cameras
        )
    
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Forward pass completed.")
    
    # === 選擇反向追蹤的初始 bbox ===
    # 而不是使用 metadata 的最後一幀，改用正向追蹤末尾分數較高的結果
    logger.info("Selecting best bbox from forward tracking for backward init...")
    
    forward_bboxes = forward_data["bboxes"]
    forward_confidences = forward_data["confidences"]
    search_window = min(30, max_safe_idx)  # 搜索最後 30 幀
    
    best_score = 0.0
    best_frame_idx = -1
    best_bbox = None
    
    for i in range(max_safe_idx - search_window, max_safe_idx):
        if i >= 0 and forward_bboxes[i] is not None and forward_confidences[i] > best_score:
            best_score = forward_confidences[i]
            best_bbox = forward_bboxes[i]
            best_frame_idx = i
    
    if best_bbox is not None:
        backward_init_bbox = best_bbox
        logger.info(f"  Selected frame {best_frame_idx} (confidence={best_score:.4f})")
    else:
        logger.warning("  No valid forward tracking result found in last {search_window} frames, using metadata bbox")
        backward_init_bbox = final_bbox
    
    # === 反向追蹤 ===
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_frames_dir, offload_video_to_cpu=True,
                                    offload_state_to_cpu=True, async_loading_frames=True)
        backward_data = run_tracking_pass(
            predictor, state, frame_names, video_frames_dir,
            max_safe_idx, backward_init_bbox, direction="backward", logger=logger,
            fusion_engine=fusion, cameras=all_cameras
        )
    
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Backward pass completed.")
    
    # === 軌跡融合 ===
    logger.info("Fusing trajectories...")
    
    fused_data = fusion.fuse_trajectories(forward_data, backward_data, max_safe_idx)
    logger.info("Trajectory fusion completed.")
    
    # === 場景切換優化 ===
    camera_switches = forward_data.get("camera_switches", [])
    if camera_switches:
        logger.info(f"Adjusting confidence near {len(camera_switches)} camera switches...")
        fused_data["confidences"] = fusion.adjust_confidence_near_switches(
            fused_data["confidences"], camera_switches
        )
        logger.info("  Confidence adjustment completed (利用 HiM2SAM 長期記憶適應期)")
    
    # === 移除速度異常 ===
    logger.info("Removing velocity outliers...")
    cleaned_bboxes = fusion.remove_outliers_velocity(fused_data["bboxes"], max_velocity=100)
    outlier_count = sum(1 for i in range(len(fused_data["bboxes"])) 
                        if fused_data["bboxes"][i] is not None and cleaned_bboxes[i] is None)
    logger.info(f"  Removed {outlier_count} outliers")
    
    # === 內插缺失的 bbox ===
    logger.info("Interpolating missing boxes...")
    interpolated_bboxes = fusion.interpolate_missing_boxes(
        cleaned_bboxes, fused_data["confidences"], max_gap=10
    )
    logger.info("Interpolation completed.")
    
    # === 平滑軌跡 ===
    logger.info("Smoothing trajectory...")
    smoothed_bboxes = fusion.smooth_trajectory(interpolated_bboxes, sigma=1.5)
    logger.info("Smoothing completed.")
    
    # === 輸出結果 ===
    logger.info("Writing output video and bbox file...")
    pbar = tqdm.tqdm(total=max_safe_idx, desc="Writing Results", unit="frame", dynamic_ncols=True)
    
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
            
            bbox_file.write(f"{x_min},{y_min},{bbox_w},{bbox_h}\n")
            
            # 繪製 bbox
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)
            
            # 繪製 mask 區域的半透明覆蓋
            mask = np.zeros((height, width), dtype=bool)
            mask[int(y_min):int(y_max), int(x_min):int(x_max)] = True
            mask_overlay = np.zeros((height, width, 3), np.uint8)
            mask_overlay[mask] = color
            img = cv2.addWeighted(img, 1, mask_overlay, 0.2, 0)
        else:
            bbox_file.write("0,0,0,0\n")
        
        out.write(img)
        pbar.update(1)
    
    pbar.close()

    # 清理
    out.release()
    bbox_file.close()
    logger.info(f"推論完成！")
    logger.info(f"影片已儲存至: {output_video_path}")
    logger.info(f"BBox 結果已儲存至: {bbox_output_path}")
    logger.info("=" * 40)
    
    del predictor
    del state
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
