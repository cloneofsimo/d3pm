export WORLD_SIZE=1 #$(nvidia-smi -L | wc -l)
deepspeed --num_gpus $WORLD_SIZE lm_deepspeed.py --learning_rate 1e-4 --run_name "test"