CUDA_VISIBLE_DEVICES=0 python scripts/run_skitb_HiM2SAM.py \
  --data_root /ssd6/ron/SkiTB/FS \
  --model_path /ssd6/ron/HiM2SAM/checkpoints/sam2.1_hiera_large.pt \
  --output_root /ssd6/ron/SkiTB-baseline \
  --save_to_video

CUDA_VISIBLE_DEVICES=0 python scripts/run_skitb_HiM2SAM.py \
  --data_root /ssd6/ron/SkiTB/JP \
  --model_path /ssd6/ron/HiM2SAM/checkpoints/sam2.1_hiera_large.pt \
  --output_root /ssd6/ron/SkiTB-baseline \
  --save_to_video

CUDA_VISIBLE_DEVICES=0 python scripts/run_skitb_HiM2SAM.py \
  --data_root /ssd6/ron/SkiTB/AL \
  --model_path /ssd6/ron/HiM2SAM/checkpoints/sam2.1_hiera_large.pt \
  --output_root /ssd6/ron/SkiTB-baseline \
  --save_to_video