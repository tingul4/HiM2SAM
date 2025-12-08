# CUDA_VISIBLE_DEVICE=2 python scripts/run_skitb_inference.py
import cv2
import gc
import numpy as np
import os
import os.path as osp
import torch
import sys

# 取得專案根目錄，以便構建相對路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 移除 project_root 從 sys.path 以避免 sam2 導入衝突
if project_root in sys.path:
    sys.path.remove(project_root)

from sam2.build_sam import build_sam2_video_predictor
import tqdm 
import shutil
from PIL import Image
import logging
from datetime import datetime

# Configure logging
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "inference.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicated console lines when the logger is reused (propagation or repeated setup)
    if logger.handlers:
        logger.handlers.clear()
    logger.propagate = False

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Stream handler (for console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    
    return logger

def load_boxes(gt_path):
    """
    讀取 boxes.txt 的所有行。
    返回一个列表，每个元素是 [x, y, w, h] (XYWH 格式)
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"找不到 bbox 檔案: {gt_path}")
        
    boxes = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if ',' in line:
                parts = line.split(',')
            else:
                parts = line.split()
            
            try:
                x, y, w, h = map(float, parts)
                boxes.append([x, y, w, h])
            except ValueError:
                pass # Skip invalid lines
    return boxes

def load_cameras(cam_path):
    """
    读取 cameras.txt。
    返回一个列表，每个元素是 camera ID (int or string)
    """
    if not os.path.exists(cam_path):
        raise FileNotFoundError(f"找不到 cameras 檔案: {cam_path}")
    
    cameras = []
    with open(cam_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            cameras.append(line)
    return cameras

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x+w, y+h]

# --- 設定路徑 ---
# 指向 test-data 裡的 metadata
metadata_dir = os.path.join(project_root, "test-data/AL0098/MC")
cameras_path = os.path.join(metadata_dir, "cameras.txt")
boxes_path = os.path.join(metadata_dir, "boxes.txt")

# Video frames path
video_frames_dir = "/ssd6/ron/SkiTB/AL/AL0098/frames"

# 輸出設定
exp_name = "skitb_inference"
timestamp = datetime.now().strftime("%m%d%H%M%S") # 精確到秒避免重複
log_parent_dir = os.path.join(project_root, "logs")
output_dir = os.path.join(log_parent_dir, f"{timestamp}_{exp_name}")
os.makedirs(output_dir, exist_ok=True) # Ensure the log directory exists before logging setup

logger = setup_logging(output_dir) # Setup logging with the new output_dir

output_video_path = os.path.join(output_dir, f"result_{timestamp}.mp4")
bbox_output_path = os.path.join(output_dir, f"bbox_{timestamp}.txt")


# 模型設定 (沿用 HiM2SAM 設定)
model_name = "large" # 使用 large 模型
checkpoint_path = os.path.join(project_root, "checkpoints", "sam2.1_hiera_large.pt")
model_cfg = "configs/him2sam/lasot/sam2.1_hiera_l.yaml" # 使用專案指定的 config

# 顏色設定 (BBox)
color = (0, 255, 0) # Green



# ReID Configuration
reid_model_path = os.path.join(os.path.dirname(project_root), 'gta-link', 'reid_checkpoints', 'sports_model.pth.tar-60')
reid_device = "cuda" if torch.cuda.is_available() else "cpu"

class ReIDMatcher:
    def __init__(self, model_path, device='cuda'):
        if FeatureExtractor is None:
            self.extractor = None
            return
            
        self.extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=model_path,
            device=device
        )
        self.gallery = []
        self.gallery_max_size = 50
        
        self.transform = T.Compose([
            T.Resize([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = device
        self.mask_generator = None # Will be set later

    def set_mask_generator(self, mask_generator):
        self.mask_generator = mask_generator

    def update_gallery(self, img, bbox):
        if self.extractor is None: return
        
        # bbox: [x, y, w, h]
        x, y, w, h = bbox
        # Ensure bbox is within image
        img_w, img_h = img.size
        x = max(0, int(x))
        y = max(0, int(y))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0: return

        crop = img.crop((x, y, x+w, y+h)).convert('RGB')
        crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature = self.extractor(crop_tensor)
            feature = feature.cpu().detach().numpy()
            feature /= np.linalg.norm(feature)
        
        self.gallery.append(feature)
        if len(self.gallery) > self.gallery_max_size:
            self.gallery.pop(0)

    def find_best_match(self, img):
        if self.extractor is None or not self.gallery or self.mask_generator is None:
            return None, 0.0

        # Generate masks
        # Convert PIL to numpy for SAM2
        img_np = np.array(img)
        masks = self.mask_generator.generate(img_np)
        
        if not masks:
            return None, 0.0
            
        # Prepare batch
        crops = []
        valid_indices = []
        
        img_w, img_h = img.size
        
        for i, mask_data in enumerate(masks):
            x, y, w, h = mask_data['bbox']
            # Filter only very small boxes (allow down to 5x5 for small/blurry objects)
            if w < 5 or h < 5: continue
            
            x = max(0, int(x))
            y = max(0, int(y))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            crop = img.crop((x, y, x+w, y+h)).convert('RGB')
            crops.append(self.transform(crop))
            valid_indices.append(i)
            
        if not crops:
            return None, 0.0
            
        batch = torch.stack(crops).to(self.device)
        
        with torch.no_grad():
            features = self.extractor(batch)
            features = features.cpu().detach().numpy()
            # Normalize
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / (norms + 1e-6)
            
        # Compare with gallery
        # Gallery: [N, D], Features: [M, D]
        gallery_feats = np.concatenate(self.gallery, axis=0) # [N, D]
        
        # Sim matrix: [M, N]
        sim_matrix = np.dot(features, gallery_feats.T)
        
        # Score for each candidate: use max similarity for robust matching
        # even when gallery might have appearance variations from different frames
        scores = np.max(sim_matrix, axis=1)
        
        best_idx_in_batch = np.argmax(scores)
        best_score = scores[best_idx_in_batch]
        
        best_mask_idx = valid_indices[best_idx_in_batch]
        best_bbox = masks[best_mask_idx]['bbox'] # xywh
        
        # Convert to xyxy
        x, y, w, h = best_bbox
        return [x, y, x+w, y+h], best_score

# 檢查 GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

def main():
    # 1. 準備影像列表
    if not os.path.exists(video_frames_dir):
        logger.error(f"錯誤: 找不到影像目錄 {video_frames_dir}")
        return

    frame_names = sorted([p for p in os.listdir(video_frames_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]])
    if not frame_names:
        logger.error("目錄中沒有找到影像檔")
        return
        
    num_frames = len(frame_names)
    logger.info(f"找到 {num_frames} 張影像")
    # 讀取第一張圖以獲取尺寸
    first_img_path = os.path.join(video_frames_dir, frame_names[0])
    first_img = cv2.imread(first_img_path)
    height, width = first_img.shape[:2]

    # 2. 載入 metadata (Cameras & Boxes)
    try:
        all_boxes = load_boxes(boxes_path)
        all_cameras = load_cameras(cameras_path)
        
        if len(all_boxes) == 0:
             logger.error("Boxes file is empty.")
             return
             
        # Simple check to ensure length matches (warn if not)
        if len(all_boxes) != num_frames or len(all_cameras) != num_frames:
            logger.warning(f"Data length mismatch! Frames: {num_frames}, Boxes: {len(all_boxes)}, Cameras: {len(all_cameras)}")
            # We will use min length to avoid index errors
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

    # Initialize ReID
    logger.info("Initializing ReID Matcher...")
    reid_matcher = ReIDMatcher(reid_model_path, device=device)
    
    # Initialize Mask Generator
    logger.info("Initializing SAM2 Automatic Mask Generator...")
    # We use the same model instance. 
    # Note: SAM2AutomaticMaskGenerator creates its own SAM2ImagePredictor which wraps the model.
    mask_generator = SAM2AutomaticMaskGenerator(
        model=predictor,
        points_per_side=32,
        pred_iou_thresh=0.7, # Slightly lower to catch more candidates
        stability_score_thresh=0.9,
        min_mask_region_area=100 # Filter very small noise
    )
    reid_matcher.set_mask_generator(mask_generator)

    # --- Log Key Parameters for Debugging ---
    logger.info("=" * 30)
    logger.info("       Key Model Parameters       ")
    logger.info("=" * 30)
    logger.info(f"num_maskmem (Total Memory): {getattr(predictor, 'num_maskmem', 'N/A')}")
    logger.info(f"rvcot_mem_long_len (Long-term): {getattr(predictor, 'rvcot_mem_long_len', 'N/A')}")
    logger.info(f"rvcot_mem_selection_method: {getattr(predictor, 'rvcot_mem_selection_method', 'N/A')}")
    logger.info(f"rvcot_ious_threshold: {getattr(predictor, 'rvcot_ious_threshold', 'N/A')}")
    logger.info(f"memory_bank_iou_threshold: {getattr(predictor, 'memory_bank_iou_threshold', 'N/A')}")
    logger.info("=" * 30)

    # 4. 初始化 Video Writer 和 BBox 檔案
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
    bbox_file = open(bbox_output_path, 'w')

    # 5. 推論迴圈
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pbar = tqdm.tqdm(total=max_safe_idx, desc="Frames", unit="frame", dynamic_ncols=True)
        # 初始化狀態
        logger.info("初始化視訊狀態...")
        state = predictor.init_state(video_frames_dir, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)

        # 加入第一幀的提示 (Frame 0)
        # obj_id=1 (追蹤物件 ID)
        _, _, _ = predictor.add_new_points_or_box(state, box=init_bbox, frame_idx=0, obj_id=1)
        
        # Update gallery with initial box
        init_img_path = os.path.join(video_frames_dir, frame_names[0])
        init_img_pil = Image.open(init_img_path).convert("RGB")
        # init_bbox is xyxy, convert to xywh for update_gallery
        init_x, init_y, init_x2, init_y2 = init_bbox
        reid_matcher.update_gallery(init_img_pil, [init_x, init_y, init_x2-init_x, init_y2-init_y])

        logger.info("開始傳播 (Propagate)...")
        
        current_proc_frame = 0
        last_bbox_xyxy = None
        
        # Use a while loop to allow restarting propagation on camera switch
        while current_proc_frame < max_safe_idx:
            
            # Start/Resume propagation from current_proc_frame
            # Note: propagate_in_video will yield results for frames starting from start_frame_idx
            
            prop_gen = predictor.propagate_in_video(state, start_frame_idx=current_proc_frame)
            
            for frame_idx, object_ids, masks in prop_gen:
                
                # --- 1. Draw & Save Result for CURRENT frame ---
                img_path = os.path.join(video_frames_dir, frame_names[frame_idx])
                img = cv2.imread(img_path)
                
                if img is None:
                    logger.warning(f"警告: 無法讀取影像 {img_path}")
                    break

                # 處理每個物件的 Mask
                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy()
                    mask = mask > 0.0 # 二值化
                    
                    # 從 Mask 計算 BBox
                    non_zero_indices = np.argwhere(mask)
                    if len(non_zero_indices) > 0:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        
                        # 將 BBox 資訊寫入 bbox.txt (格式: x,y,w,h)
                        bbox_x = x_min
                        bbox_y = y_min
                        bbox_w = x_max - x_min
                        bbox_h = y_max - y_min
                        bbox_file.write(f"{bbox_x},{bbox_y},{bbox_w},{bbox_h}\n")

                        last_bbox_xyxy = [x_min, y_min, x_max, y_max]
                        
                        # Update ReID gallery (every frame might be too much, maybe every 5 frames?)
                        # But we want to capture the appearance change.
                        # Let's do every frame for now, gallery size is limited anyway.
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        reid_matcher.update_gallery(pil_img, [bbox_x, bbox_y, bbox_w, bbox_h])

                        # 繪製 BBox (x_min, y_min, x_max, y_max)
                        # cv2.rectangle 需要 (x, y), (x+w, y+h)
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)
                        
                        # 選擇性：也可以繪製半透明 Mask
                        mask_overlay = np.zeros((height, width, 3), np.uint8)
                        mask_overlay[mask] = color
                        img = cv2.addWeighted(img, 1, mask_overlay, 0.4, 0)
                    else:
                        # 如果沒有檢測到 Mask，則寫入一個空行或 (0,0,0,0)
                        bbox_file.write("0,0,0,0\n")


                # 寫入影片
                out.write(img)
                
                # Update progress bar
                pbar.update(1)

                # --- 2. Check for Camera Switch in NEXT frame ---
                next_frame_idx = frame_idx + 1
                
                if next_frame_idx < max_safe_idx:
                    curr_cam = all_cameras[frame_idx]
                    next_cam = all_cameras[next_frame_idx]
                    
                    if next_cam != curr_cam:
                        logger.info(f"[Switch Detected] Frame {frame_idx} (Cam {curr_cam}) -> Frame {next_frame_idx} (Cam {next_cam})")
                        logger.info(f"Resetting state and re-initializing with ReID search for frame {next_frame_idx}...")
                        
                        # Reset memory
                        predictor.reset_state(state)
                        
                        # Search for object in new camera view
                        found_new_init = False
                        search_frame_idx = next_frame_idx
                        
                        # Look ahead up to 60 frames (2 seconds) to find a good initialization
                        look_ahead_limit = min(max_safe_idx, next_frame_idx + 60)
                        best_init_frame = -1
                        best_init_box = None
                        
                        # First pass: Find the first valid detection
                        logger.info(f"ReID gallery size: {len(reid_matcher.gallery)} samples")
                        for temp_idx in range(next_frame_idx, look_ahead_limit):
                            temp_img_path = os.path.join(video_frames_dir, frame_names[temp_idx])
                            temp_img_pil = Image.open(temp_img_path).convert("RGB")
                            
                            # Use ReID to find best match in this frame
                            temp_box, score = reid_matcher.find_best_match(temp_img_pil)
                            
                            logger.info(f"Frame {temp_idx}: ReID score = {score:.4f}, box = {temp_box}")
                            
                            # Lower threshold to 0.5 for better recall after camera switch
                            if temp_box is not None and score > 0.5:
                                best_init_frame = temp_idx
                                best_init_box = temp_box
                                logger.info(f"Found valid initialization at frame {temp_idx} with score {score:.4f}")
                                break
                        
                        if best_init_frame != -1:
                            # Initialize at the found frame
                            _, _, _ = predictor.add_new_points_or_box(state, box=best_init_box, frame_idx=best_init_frame, obj_id=1)
                            found_new_init = True
                            
                            # If we skipped frames, we need to track BACKWARDS to fill the gap
                            if best_init_frame > next_frame_idx:
                                logger.info(f"Tracking BACKWARDS from {best_init_frame} to {next_frame_idx}...")
                                # Propagate backwards
                                # Note: propagate_in_video yields frames in reverse order when reverse=True
                                back_gen = predictor.propagate_in_video(state, start_frame_idx=best_init_frame, max_frame_num_to_track=(best_init_frame - next_frame_idx), reverse=True)
                                
                                # We need to store backward results to write them in correct order later?
                                # Actually, since we are writing sequentially to a video file, we can't easily go back.
                                # BUT, the current loop structure expects us to write frames sequentially.
                                # So we must buffer the backward results and write them NOW before continuing forward.
                                
                                backward_results = {} # frame_idx -> (img, bbox_str)
                                
                                for b_frame_idx, b_obj_ids, b_masks in back_gen:
                                    # Process backward frame
                                    b_img_path = os.path.join(video_frames_dir, frame_names[b_frame_idx])
                                    b_img = cv2.imread(b_img_path)
                                    
                                    if b_img is None:
                                        continue
                                    
                                    # Assuming single object
                                    b_mask = b_masks[0][0].cpu().numpy() > 0.0
                                    non_zero = np.argwhere(b_mask)
                                    
                                    if len(non_zero) > 0:
                                        y_min, x_min = non_zero.min(axis=0).tolist()
                                        y_max, x_max = non_zero.max(axis=0).tolist()
                                        bbox_str = f"{x_min},{y_min},{x_max-x_min},{y_max-y_min}\n"
                                        cv2.rectangle(b_img, (x_min, y_min), (x_max, y_max), color, 3)
                                        mask_overlay = np.zeros((height, width, 3), np.uint8)
                                        mask_overlay[b_mask] = color
                                        b_img = cv2.addWeighted(b_img, 1, mask_overlay, 0.4, 0)
                                        
                                        # Update gallery during backward tracking too?
                                        # Maybe safer not to, as backward tracking might be less stable?
                                        # But it helps bridge the gap. Let's do it.
                                        img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
                                        pil_img = Image.fromarray(img_rgb)
                                        reid_matcher.update_gallery(pil_img, [x_min, y_min, x_max-x_min, y_max-y_min])
                                        
                                    else:
                                        bbox_str = "0,0,0,0\n"
                                    
                                    backward_results[b_frame_idx] = (b_img, bbox_str)
                                
                                # Now write the gap frames (next_frame_idx to best_init_frame - 1)
                                for gap_idx in range(next_frame_idx, best_init_frame):
                                    if gap_idx in backward_results:
                                        gap_img, gap_bbox = backward_results[gap_idx]
                                        bbox_file.write(gap_bbox)
                                        out.write(gap_img)
                                    else:
                                        # If backward tracking failed for this frame, write empty
                                        bbox_file.write("0,0,0,0\n")
                                        gap_img = cv2.imread(os.path.join(video_frames_dir, frame_names[gap_idx]))
                                        out.write(gap_img)
                                    pbar.update(1)

                            # Now set current_proc_frame to best_init_frame to start forward tracking
                            current_proc_frame = best_init_frame
                            
                        else:
                            # If still not found after look-ahead, we might be in trouble.
                            # Just continue searching frame-by-frame as before (or give up for this shot)
                            logger.warning("Could not find object in next 60 frames. Continuing sequential search...")
                            
                            # Fallback to the sequential search logic (which writes empty frames)
                            while search_frame_idx < max_safe_idx:
                                search_img_path = os.path.join(video_frames_dir, frame_names[search_frame_idx])
                                temp_img_pil = Image.open(search_img_path).convert("RGB")
                                new_box, score = reid_matcher.find_best_match(temp_img_pil)
                                
                                logger.info(f"Sequential search frame {search_frame_idx}: ReID score = {score:.4f}")
                                
                                # Lower threshold even more in sequential search (0.45)
                                if new_box is not None and score > 0.45:
                                    logger.info(f"Found object at frame {search_frame_idx} with score {score:.4f}")
                                    _, _, _ = predictor.add_new_points_or_box(state, box=new_box, frame_idx=search_frame_idx, obj_id=1)
                                    current_proc_frame = search_frame_idx
                                    found_new_init = True
                                    break
                                else:
                                    bbox_file.write("0,0,0,0\n")
                                    img_search = cv2.imread(search_img_path)
                                    out.write(img_search)
                                    pbar.update(1)
                                    search_frame_idx += 1
                            
                            if not found_new_init:
                                current_proc_frame = max_safe_idx

                        break # Break inner for-loop, outer while-loop will restart propagation
                
                # Normal increment if no switch or end of video
                current_proc_frame = next_frame_idx
            
            # If the generator finished naturally (end of video), break the while loop
            if current_proc_frame >= max_safe_idx:
                break

        pbar.close()

    # 6. 清理
    out.release()
    bbox_file.close() # 關閉 bbox.txt 檔案
    logger.info(f"推論完成！影片已儲存至: {output_video_path}")
    logger.info(f"BBox 結果已儲存至: {bbox_output_path}")
    
    # 釋放資源
    del predictor
    del state
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()