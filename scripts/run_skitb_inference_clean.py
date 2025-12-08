# Simple HiM2SAM Inference without ReID
# CUDA_VISIBLE_DEVICE=2 python scripts/run_skitb_inference_clean.py
import cv2
import gc
import numpy as np
import os
import torch
import sys
import tqdm
import logging
from datetime import datetime

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
    """讀取 boxes.txt 的所有行，返回一個列表"""
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

# 設定路徑
metadata_dir = os.path.join(project_root, "test-data/AL0098/MC")
cameras_path = os.path.join(metadata_dir, "cameras.txt")
boxes_path = os.path.join(metadata_dir, "boxes.txt")
video_frames_dir = "/ssd6/ron/SkiTB/AL/AL0098/frames"

# 輸出設定
exp_name = "skitb_inference_clean"
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

    logger.info("=" * 30)
    logger.info("       Key Model Parameters       ")
    logger.info("=" * 30)
    logger.info(f"num_maskmem (Total Memory): {getattr(predictor, 'num_maskmem', 'N/A')}")
    logger.info(f"rvcot_mem_long_len (Long-term): {getattr(predictor, 'rvcot_mem_long_len', 'N/A')}")
    logger.info(f"rvcot_mem_selection_method: {getattr(predictor, 'rvcot_mem_selection_method', 'N/A')}")
    logger.info("=" * 30)

    # 4. 初始化 Video Writer 和 BBox 檔案
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
    bbox_file = open(bbox_output_path, 'w')

    # 5. 推論迴圈
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pbar = tqdm.tqdm(total=max_safe_idx, desc="Frames", unit="frame", dynamic_ncols=True)
        
        logger.info("初始化視訊狀態...")
        state = predictor.init_state(video_frames_dir, offload_video_to_cpu=True, 
                                    offload_state_to_cpu=True, async_loading_frames=True)

        # 加入第一幀的提示
        _, _, _ = predictor.add_new_points_or_box(state, box=init_bbox, frame_idx=0, obj_id=1)

        logger.info("開始傳播 (Propagate)...")
        
        prop_gen = predictor.propagate_in_video(state, start_frame_idx=0)
        
        for frame_idx, object_ids, masks in prop_gen:
            
            # 讀取影像
            img_path = os.path.join(video_frames_dir, frame_names[frame_idx])
            img = cv2.imread(img_path)
            
            if img is None:
                logger.warning(f"警告: 無法讀取影像 {img_path}")
                break

            # 處理 Mask
            mask_found = False
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                
                # 從 Mask 計算 BBox
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) > 0:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    
                    bbox_x = x_min
                    bbox_y = y_min
                    bbox_w = x_max - x_min
                    bbox_h = y_max - y_min
                    
                    bbox_file.write(f"{bbox_x},{bbox_y},{bbox_w},{bbox_h}\n")
                    mask_found = True
                    
                    # 繪製 BBox
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)
                    
                    # 繪製 Mask
                    mask_overlay = np.zeros((height, width, 3), np.uint8)
                    mask_overlay[mask] = color
                    img = cv2.addWeighted(img, 1, mask_overlay, 0.4, 0)
            
            if not mask_found:
                bbox_file.write("0,0,0,0\n")

            # 寫入影片
            out.write(img)
            pbar.update(1)

        pbar.close()

    # 6. 清理
    out.release()
    bbox_file.close()
    logger.info(f"推論完成！影片已儲存至: {output_video_path}")
    logger.info(f"BBox 結果已儲存至: {bbox_output_path}")
    
    del predictor
    del state
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
