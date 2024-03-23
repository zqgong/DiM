CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu --num_processes 2 --main_process_port 29500 --mixed_precision fp16 train.py --model DiT-B-MOE/4 --global-batch-size 1024
 
