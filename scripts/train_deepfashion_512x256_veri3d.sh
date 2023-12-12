MASTER_PORT=$((12000 + $RANDOM % 20000))
NUM_GPU=8
set -x

python -m torch.distributed.launch --nproc_per_node ${NUM_GPU} --master_port=${MASTER_PORT} train_deepfashion.py --batch 1 --chunk 1 --expname train_deepfashion_512x256_veri3d --dataset_path datasets/DeepFashion --style_dim 256 --renderer_spatial_output_dim 512 256 --white_bg --r1 300 --random_flip --small_aug --iter 1000000 --adjust_gamma --gamma_lb 20  --gaussian_weighted_sampler --sampler_std 15 --N_samples 28 --glr 1e-4
