export CUDA_VISIBLE_DEVICES=4,5,6,7

uv run torchrun --standalone --nproc-per-node 4 --nnodes 1 train_ddp_v2.py --per_device_batch_size 8