import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import glob
import json
import logging
sys.path.append("./sam2")
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor
import csv
import ast
import re

color = [(255, 0, 0)]

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

def load_segments_to_dict(csv_path):
    """將 CSV 載入為巢狀字典結構 - 手動解析避免 bbox 內逗號問題"""
    segments_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳過 header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # 使用正則找出兩個 bbox (格式: [x, y, w, h])
        # 模式: discipline,seq,segment,start,end,best_frame,[...],[...],conf,iou,score
        bbox_pattern = r'\[([^\]]+)\]'
        bboxes = re.findall(bbox_pattern, line)
        
        if len(bboxes) != 2:
            print(f"Warning: Could not find 2 bboxes in line: {line[:100]}")
            continue
        
        # 移除所有 bbox 後分割其他欄位
        line_without_bbox = re.sub(bbox_pattern, '||BBOX||', line)
        parts = line_without_bbox.split(',')
        
        # 過濾出非 bbox 的部分
        clean_parts = [p for p in parts if p != '||BBOX||']
        
        if len(clean_parts) < 9:  # discipline,seq,segment,start,end,best_frame,conf,iou,score
            print(f"Warning: Not enough fields in line: {line[:100]}")
            continue
        
        try:
            discipline = clean_parts[0]
            seq = clean_parts[1].strip()
            segment_id = int(clean_parts[2])
            start = int(clean_parts[3])
            end = int(clean_parts[4])
            best_frame = int(clean_parts[5])
            conf = float(clean_parts[6])
            iou = float(clean_parts[7])
            score = float(clean_parts[8])
            
            # 解析 bbox
            him_bbox = [float(x.strip()) for x in bboxes[0].split(',')]
            rf_bbox = [float(x.strip()) for x in bboxes[1].split(',')]
            
            # 初始化序列
            if seq not in segments_dict:
                segments_dict[seq] = {}
            
            segments_dict[seq][segment_id] = {
                'start': start,
                'end': end,
                'best_frame': best_frame,
                'him_bbox': him_bbox,
                'rf_bbox': rf_bbox,
                # 'conf': conf,
                # 'iou': iou,
                # 'score': score
            }
        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {e}")
            print(f"  Line: {line[:150]}")
            continue
    
    return segments_dict

def load_txt(gt_path):
    """Load only the FIRST bbox (x,y,w,h)"""
    with open(gt_path, 'r') as f:
        line = f.readline().strip()
    x, y, w, h = map(float, line.split(','))
    x, y, w, h = int(x), int(y), int(w), int(h)
    return (x, y, x + w, y + h), 0    # convert to x1,y1,x2,y2

def load_last_txt(gt_path):
    """Load only the LAST bbox (x,y,w,h)"""
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        # Filter out empty lines
        lines = [line.strip() for line in lines if line.strip()]
        if not lines:
            raise ValueError(f"File {gt_path} is empty or contains only whitespace.")
        last_line = lines[-1]
        
    x, y, w, h = map(float, last_line.split(','))
    x, y, w, h = int(x), int(y), int(w), int(h)
    return (x, y, x + w, y + h), 0    # convert to x1,y1,x2,y2

def bbox_xywh_to_xyxy(bbox):
    """Convert bbox from [x, y, w, h] format to (x1, y1, x2, y2) format
    
    Args:
        bbox: list or tuple of [x, y, w, h] format
        
    Returns:
        tuple: ((x1, y1, x2, y2), 0)
    """
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    return (x, y, x + w, y + h), 0

def save_frame_with_bbox(frames_dir, frame_idx, bbox, output_path, color=(0, 255, 0), thickness=2):
    """Save a frame with bounding box drawn on it
    
    Args:
        frames_dir: directory containing frame images
        frame_idx: frame index to load (0-based)
        bbox: bounding box in format (x1, y1, x2, y2) or [x, y, w, h]
        output_path: path to save the output image
        color: BGR color tuple for bbox (default green)
        thickness: line thickness for bbox (default 2)
    """
    # Get sorted list of frame files
    frame_names = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ])
    
    if frame_idx >= len(frame_names):
        raise ValueError(f"Frame index {frame_idx} out of range (total: {len(frame_names)})")
    
    # Load the frame
    frame_path = osp.join(frames_dir, frame_names[frame_idx])
    img = cv2.imread(frame_path)
    
    if img is None:
        raise ValueError(f"Failed to load image: {frame_path}")
    
    # Convert bbox format if needed (assume it's [x, y, w, h] if length is 4 and values seem off)
    if len(bbox) == 4:
        x, y, x2, y2 = bbox
        # Check if this looks like (x, y, w, h) format - if x2 < x, assume w/h
        if x2 < x or y2 < y:
            w, h = x2, y2
            x2, y2 = x + w, y + h
        x1, y1 = int(x), int(y)
        x2, y2 = int(x2), int(y2)
    else:
        raise ValueError("Bbox must have 4 values")
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Save image
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Saved frame with bbox to: {output_path}")

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/him2sam/lasot/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/him2sam/lasotext/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/him2sam/lasotext/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/him2sam/lasotext/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def process_sequence(predictor, seq_name, args, logger, seg_data):
    # Setup output directory
    save_dir = osp.join(args.output_root, seq_name)
    os.makedirs(save_dir, exist_ok=True)
    seq_path = osp.join(args.data_root, seq_name)
    frames_dir = osp.join(seq_path, "frames")
    
    if not osp.isdir(frames_dir):
        return

    # Output paths
    txt_output_path = osp.join(save_dir, f"{seq_name}_track.txt")
    video_output_path = osp.join(save_dir, f"{seq_name}_demo.mp4")
    
    logger.info(f"Processing sequence: {seq_name}")

    # 1. Collect all predictions
    sequence_predictions = {} # absolute_frame_idx -> [x,y,w,h]
    
    # Sort segments by ID to ensure logical processing order (though we store by frame_idx)
    sorted_seg_ids = sorted(seg_data.keys())
    
    for seg_id in sorted_seg_ids:
        info = seg_data[seg_id]
        start, end = info['start'], info['end']
        best = info['best_frame']
        bbox = info['him_bbox'] # [x,y,w,h]
        
        # Convert bbox to xyxy for SAM2
        bbox_xyxy, _ = bbox_xywh_to_xyxy(bbox)
        
        # --- Reverse Tracking (best -> start) ---
        if best > start:
            # We want frames from 'best' down to 'start' (inclusive)
            start_arg = best
            target_end = start
            # Slice logic: [start : end : -1]
            # If we want to include index 0, end must be None
            if target_end == 0:
                end_arg = None
            else:
                end_arg = target_end - 1
            
            # Init state
            state = predictor.init_state(frames_dir, offload_video_to_cpu=True, reverse=False, start=start_arg, end=end_arg, is_reverse=-1)
            
            # Add init box at relative index 0 (which is 'best' frame)
            predictor.add_new_points_or_box(state, box=bbox_xyxy, frame_idx=0, obj_id=0)
            
            # Propagate
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                # frame_idx is relative (0, 1, 2...)
                # real frame is start_arg - frame_idx
                abs_frame_idx = start_arg - frame_idx
                
                mask = masks[0][0].cpu().numpy() > 0.0
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    pred_box = [0, 0, 0, 0]
                else:
                    x1, y1 = xs.min(), ys.min()
                    x2, y2 = xs.max(), ys.max()
                    pred_box = [x1, y1, x2 - x1, y2 - y1] # xywh
                
                sequence_predictions[abs_frame_idx] = pred_box
            
            del state
            gc.collect()
            torch.cuda.empty_cache()

        # --- Forward Tracking (best -> end) ---
        if best < end:
            # We want frames from 'best' up to 'end' (inclusive)
            start_arg = best
            target_end = end
            end_arg = target_end + 1
            
            state = predictor.init_state(frames_dir, offload_video_to_cpu=True, reverse=False, start=start_arg, end=end_arg, is_reverse=1)
            
            predictor.add_new_points_or_box(state, box=bbox_xyxy, frame_idx=0, obj_id=0)
            
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                # frame_idx is relative (0, 1, 2...)
                # real frame is start_arg + frame_idx
                abs_frame_idx = start_arg + frame_idx
                
                mask = masks[0][0].cpu().numpy() > 0.0
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    pred_box = [0, 0, 0, 0]
                else:
                    x1, y1 = xs.min(), ys.min()
                    x2, y2 = xs.max(), ys.max()
                    pred_box = [x1, y1, x2 - x1, y2 - y1]
                
                sequence_predictions[abs_frame_idx] = pred_box
                
            del state
            gc.collect()
            torch.cuda.empty_cache()
            
        # Handle single frame case
        if start == end == best:
             sequence_predictions[best] = bbox

    # 2. Write Results
    if not sequence_predictions:
        logger.info(f"No predictions for {seq_name}")
        return

    sorted_frames = sorted(sequence_predictions.keys())
    
    # Setup VideoWriter
    all_frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])
    
    if not all_frame_files:
        return

    first_frame_idx = sorted_frames[0]
    if first_frame_idx >= len(all_frame_files):
        logger.error(f"Frame index {first_frame_idx} out of bounds.")
        return

    img_path = os.path.join(frames_dir, all_frame_files[first_frame_idx])
    h, w = cv2.imread(img_path).shape[:2]
    
    if args.save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, 30, (w, h))
    
    with open(txt_output_path, "w") as f:
        for idx in sorted_frames:
            box = sequence_predictions[idx]
            x, y, w_box, h_box = map(int, box)
            f.write(f"{x},{y},{w_box},{h_box}\n")
            
            if args.save_to_video:
                if 0 <= idx < len(all_frame_files):
                    img_path = os.path.join(frames_dir, all_frame_files[idx])
                    img = cv2.imread(img_path)
                    if img is not None:
                        cv2.rectangle(img, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)
                        # Add frame number text for debug
                        cv2.putText(img, str(idx), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        out.write(img)
    
    if args.save_to_video:
        out.release()
        logger.info(f"Saved video to {video_output_path}")

def main(args):
    # Create output root if not exists
    os.makedirs(args.output_root, exist_ok=True)
    logger = setup_logging(args.output_root)

    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # Find split json file
        json_files = glob.glob(osp.join(args.data_root, "*_train_val_test_date_60-40.json"))
        json_path = None
        if not json_files:
            logger.info(f"Warning: No JSON file ending with '_train_val_test_date_60-40.json' found in {args.data_root}. Processing all subdirectories.")
        else:
            if len(json_files) > 1:
                logger.info(f"Warning: Multiple JSON files ending with '_train_val_test_date_60-40.json' found. Using the first one: {json_files[0]}")
            json_path = json_files[0]
        
        subdirs = []
        if json_path:
            logger.info(f"Loading test sequences from {json_path}")
            with open(json_path, "r") as f:
                split_data = json.load(f)
                # Only use for filtering if the 'test' list exists and is not empty
                subdirs = split_data.get("test", [])
                if not subdirs:
                    raise Exception("Warning: 'test' list in JSON is empty. Processing all subdirectories instead.")

        logger.info(f"test sequences found: {len(subdirs)}")
        for seq_name in subdirs:
            segments = load_segments_to_dict('/ssd6/ron/good_segments_summary.csv')
            # 存取特定序列的特定片段
            seg_data = segments[seq_name]
            print(len(seg_data))
            process_sequence(predictor, seq_name, args, logger, seg_data)
            break  # For debugging, process only one sequence
            
    # Cleanup predictor at the end
    del predictor
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=False, default="/ssd6/ron/SkiTB/AL", help="Root directory containing sequence folders (e.g. test-data/AL/)")
    parser.add_argument("--output_root", required=False, default="/ssd6/ron/SkiTB-test", help="Root directory for saving results")
    parser.add_argument("--model_path", required=False, default="/ssd6/ron/HiM2SAM/checkpoints/sam2.1_hiera_large.pt", help="Path to the HiM2SAM model checkpoint")
    parser.add_argument("--save_to_video", default=True, action="store_true")
    args = parser.parse_args()
    main(args)
