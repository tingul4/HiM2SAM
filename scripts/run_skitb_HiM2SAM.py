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

def load_txt(gt_path):
    """Load only the FIRST bbox (x,y,w,h)"""
    with open(gt_path, 'r') as f:
        line = f.readline().strip()
    x, y, w, h = map(float, line.split(','))
    x, y, w, h = int(x), int(y), int(w), int(h)
    return (x, y, x + w, y + h), 0    # convert to x1,y1,x2,y2

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/him2sam/lasotext/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/him2sam/lasotext/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/him2sam/lasotext/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/him2sam/lasotext/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def process_sequence(predictor, seq_name, args, logger):
    seq_path = osp.join(args.data_root, seq_name)
    frames_dir = osp.join(seq_path, "frames")
    
    if not osp.isdir(frames_dir):
        # Only print if it looks like a sequence folder (optional, to reduce noise)
        return

    # Find init txt file
    txt_path = osp.join(seq_path, "MC", "boxes.txt")
    if not osp.isfile(txt_path):
        logger.info(f"Skipping {seq_name}: No boxes.txt found.")
        return
    
    logger.info(f"Processing sequence: {seq_name}")

    # Get frame paths
    frames = sorted([
        osp.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith(".jpg")
    ])
    
    if not frames:
        logger.info(f"Skipping {seq_name}: No frames found in {frames_dir}.")
        return

    height, width = cv2.imread(frames[0]).shape[:2]
    
    # Load init bbox
    bbox_init, track_label = load_txt(txt_path)
    
    # Setup output directory
    # If data_root is "test-data" and we process "test-data/AL0098", 
    # and output_root is "results", we want "results/AL0098".
    # Since seq_path is absolute or relative to CWD, let's just use seq_name
    # to create a subfolder in output_root.
    save_dir = osp.join(args.output_root, seq_name)
    os.makedirs(save_dir, exist_ok=True)

    # Output paths
    txt_output_path = osp.join(save_dir, f"{seq_name}_track.txt")
    video_output_path = osp.join(save_dir, f"{seq_name}_demo.mp4")

    # Optional video output
    if args.save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, 30, (width, height))
    
    predictions = []
    
    # Run inference
    # Note: predictor.init_state takes the folder containing images. 
    # Since we found images in frames_dir, we pass frames_dir.
    state = predictor.init_state(frames_dir, offload_video_to_cpu=True)
    
    # Initialize object on frame 0
    predictor.add_new_points_or_box(state, box=bbox_init, frame_idx=0, obj_id=0)
    
    # Tracking loop
    for frame_idx, object_ids, masks in tqdm(predictor.propagate_in_video(state), total=len(frames), desc=f"Tracking {seq_name}"):
        mask = masks[0][0].cpu().numpy() > 0.0
        ys, xs = np.where(mask)
        if len(xs) == 0:
            pred_box = [0, 0, 0, 0]
        else:
            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()
            pred_box = [x1, y1, x2 - x1, y2 - y1]
        
        predictions.append(pred_box)
        
        # Optional visualization
        if args.save_to_video:
            img = cv2.imread(frames[frame_idx])
            cv2.rectangle(img,
                          (pred_box[0], pred_box[1]),
                          (pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]),
                          (255, 0, 0), 2)
            out.write(img)
            
    # Save TXT results
    with open(txt_output_path, "w") as f:
        for x, y, w, h in predictions:
            f.write(f"{x},{y},{w},{h}\n")
            
    if args.save_to_video:
        out.release()
        
    # Cleanup state for this video
    del state
    gc.collect()
    torch.cuda.empty_cache()

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
            process_sequence(predictor, seq_name, args, logger)
            
    # Cleanup predictor at the end
    del predictor
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Root directory containing sequence folders (e.g. test-data/AL/)")
    parser.add_argument("--output_root", required=True, help="Root directory for saving results")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_to_video", action="store_true")
    args = parser.parse_args()
    main(args)
