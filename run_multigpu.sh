export WORLD_SIZE=$(nvidia-smi -L | wc -l)
deepspeed --num_gpus $WORLD_SIZE lm_deepspeed.py --learning_rate 1e-4