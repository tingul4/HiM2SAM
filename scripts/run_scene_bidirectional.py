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
from tqdm import tqdm

# Add sam2 to path
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "inference_scene_bidirectional.log")

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

def load_bbox_for_frame(gt_path, frame_idx):
    """
    Load the bbox (x,y,w,h) for a specific frame index from the GT file.
    Assumes one line per frame in the GT file.
    """
    if not osp.exists(gt_path):
        return None

    with open(gt_path, 'r') as f:
        lines = f.readlines()
    
    if frame_idx >= len(lines):
        # Fallback or error
        return None
    
    line = lines[frame_idx].strip()
    try:
        parts = list(map(float, line.split(',')))
    except ValueError:
        return None
    
    # Check if the line is valid (sometimes GT has NaN or empty lines for occluded objects)
    if any(np.isnan(parts)):
        return None

    x, y, w, h = parts
    # Convert to [x1, y1, x2, y2] for SAM 2
    return [x, y, x + w, y + h]

def load_cameras(cam_path):
    """Load camera IDs for each frame."""
    if not os.path.exists(cam_path):
        return None
    
    cameras = []
    with open(cam_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cameras.append(line)
    return cameras

def identify_scenes(cameras, num_frames):
    """
    Split frames into scenes based on camera ID changes.
    Returns list of (start_idx, end_idx) tuples.
    """
    scenes = []
    if not cameras or len(cameras) != num_frames:
        # Fallback: treat whole video as one scene
        scenes.append((0, num_frames - 1))
        return scenes

    current_cam = cameras[0]
    start_idx = 0
    for i, cam in enumerate(cameras):
        if cam != current_cam:
            scenes.append((start_idx, i - 1))
            current_cam = cam
            start_idx = i
    # Add the last scene
    scenes.append((start_idx, num_frames - 1))
    return scenes

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

def process_sequence(predictor, seq_name, args, logger):
    seq_path = osp.join(args.data_root, seq_name)
    frames_dir = osp.join(seq_path, "frames")
    
    if not osp.isdir(frames_dir):
        return

    # Paths
    txt_path = osp.join(seq_path, "MC", "boxes.txt")
    cam_path = osp.join(seq_path, "MC", "cameras.txt")
    
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
        logger.info(f"Skipping {seq_name}: No frames found.")
        return

    height, width = cv2.imread(frames[0]).shape[:2]
    num_frames = len(frames)

    # Load cameras and identify scenes
    cameras = load_cameras(cam_path)
    if cameras is None:
        logger.warning(f"  No cameras.txt found for {seq_name}. Treating as single scene.")
    elif len(cameras) != num_frames:
        logger.warning(f"  cameras.txt length ({len(cameras)}) matches frames ({num_frames}) mismatch. Treating as single scene.")
        cameras = None # Fallback
        
    scenes = identify_scenes(cameras, num_frames)
    logger.info(f"  Identified {len(scenes)} scenes.")

    # Initialize predictor state
    state = predictor.init_state(frames_dir, offload_video_to_cpu=True)

    # Storage for results: frame_idx -> [x, y, w, h]
    # Initialize with zeros
    all_predictions = [[0, 0, 0, 0] for _ in range(num_frames)]
    anchor_indices = []

    # Process each scene
    for scene_idx, (start_idx, end_idx) in enumerate(scenes):
        # Calculate anchor (middle frame)
        anchor_idx = start_idx + (end_idx - start_idx) // 2
        anchor_indices.append(anchor_idx)
        
        logger.info(f"    Scene {scene_idx+1}/{len(scenes)}: Frames {start_idx}-{end_idx}, Anchor: {anchor_idx}")

        # Get GT bbox for anchor
        anchor_bbox = load_bbox_for_frame(txt_path, anchor_idx)
        
        if anchor_bbox is None or anchor_bbox == [0, 0, 0, 0]:
            logger.warning(f"      Invalid or missing GT for anchor frame {anchor_idx}. Skipping scene.")
            continue

        # Reset state to clear previous scene's prompts/memory
        # This ensures we treat each camera view independently
        predictor.reset_state(state)

        # Add prompt
        predictor.add_new_points_or_box(state, box=anchor_bbox, frame_idx=anchor_idx, obj_id=0)

        # Run Bidirectional Tracking for this scene range
        scene_bboxes = predictor.propagate_in_video_bidirectional(
            state,
            start_frame_idx=start_idx,
            end_frame_idx=end_idx,
            anchor_frame_idx=anchor_idx
        )

        # Store results into global list
        for f_idx, obj_res in scene_bboxes.items():
            if 0 in obj_res: # obj_id 0
                x1, y1, x2, y2 = obj_res[0]
                # Convert to xywh
                w = x2 - x1
                h = y2 - y1
                # Simple valid check
                if w > 0 and h > 0:
                    all_predictions[f_idx] = [x1, y1, w, h]

    # Setup output directory
    save_dir = osp.join(args.output_root, seq_name)
    os.makedirs(save_dir, exist_ok=True)

    txt_output_path = osp.join(save_dir, f"{seq_name}_track_scene_bidirectional.txt")
    video_output_path = osp.join(save_dir, f"{seq_name}_demo_scene_bidirectional.mp4")

    # Save TXT
    with open(txt_output_path, "w") as f:
        for pred in all_predictions:
            f.write(f"{pred[0]},{pred[1]},{pred[2]},{pred[3]}\n")

    # Save Video
    if args.save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, 30, (width, height))
        
        for i in range(num_frames):
            img = cv2.imread(frames[i])
            x, y, w, h = all_predictions[i]
            
            # Draw bbox
            if w > 0 and h > 0:
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            
            # Draw Scene Info & Anchor marker
            if i in anchor_indices:
                cv2.putText(img, "ANCHOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(img)
        
        out.release()

    # Cleanup state
    del state
    gc.collect()
    torch.cuda.empty_cache()

def main(args):
    # Create output root
    os.makedirs(args.output_root, exist_ok=True)
    logger = setup_logging(args.output_root)

    model_cfg = determine_model_cfg(args.model_path)
    logger.info(f"Building predictor with config: {model_cfg}")
    
    # Enable rvcot_mode for HiM2SAM
    predictor = build_sam2_video_predictor(
        model_cfg, 
        args.model_path, 
        device="cuda:0",
        rvcot_mode=True
    )
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # Find split json file
        json_files = glob.glob(osp.join(args.data_root, "*_train_val_test_date_60-40.json"))
        subdirs = []
        
        if json_files:
            json_path = json_files[0]
            logger.info(f"Loading sequence list from {json_path}")
            with open(json_path, "r") as f:
                split_data = json.load(f)
                subdirs = split_data.get("test", [])
        else:
            logger.info(f"No split JSON found. Scanning {args.data_root} ...")
            all_items = os.listdir(args.data_root)
            for item in all_items:
                if osp.isdir(osp.join(args.data_root, item, "frames")):
                    subdirs.append(item)
            subdirs = sorted(subdirs)

        if not subdirs:
            logger.error("No sequences found.")
            return

        logger.info(f"Found {len(subdirs[:1])} sequences.")
        
        for seq_name in subdirs[:1]:
            try:
                process_sequence(predictor, seq_name, args, logger)
            except Exception as e:
                logger.error(f"Error processing {seq_name}: {e}")
                import traceback
                traceback.print_exc()

    del predictor
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=False, default="/ssd6/ron/SkiTB/AL", help="Root directory containing sequence folders")
    parser.add_argument("--output_root", required=False, default="/ssd6/ron/SkiTB-test-scene-bidirectional", help="Root directory for saving results")
    parser.add_argument("--model_path", required=False, default="/ssd6/ron/HiM2SAM/checkpoints/sam2.1_hiera_large.pt", help="Model checkpoint path")
    parser.add_argument("--save_to_video", default=True, action="store_true", help="Save debug video")
    
    args = parser.parse_args()
    main(args)
