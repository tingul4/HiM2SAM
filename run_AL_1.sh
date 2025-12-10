CUDA_VISIBLE_DEVICES=1 python scripts/run_him2sam_bidirectional.py \
  --data_root /ssd6/ron/SkiTB/AL \
  --model_path /ssd6/ron/HiM2SAM/checkpoints/sam2.1_hiera_large.pt \
  --output_root /ssd6/ron/SkiTB-bidirectional \
  --save_to_video \
  --start_seq 10 \
  --end_seq 20