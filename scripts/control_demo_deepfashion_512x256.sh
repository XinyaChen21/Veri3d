#!/bin/bash
python control_demo.py --batch 1 --chunk 1 --expname train_deepfashion_512x256_veri3d --dataset_path datasets/DeepFashion --style_dim 256 --renderer_spatial_output_dim 512 256 --white_bg  --N_samples 28 --ckpt 450000 --identities 1000 --truncation_ratio 1.0 --render_video --move_camera --multiple_sample

