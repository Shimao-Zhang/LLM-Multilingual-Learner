random_port=$((RANDOM % 10000 + 20000))

echo "Using random port: $random_port"

export CUDA_VISIBLE_DEVICES=2

export HF_HOME=/home/nfs03/zhangsm/data

torchrun --nproc_per_node 1 --master_port $random_port ../paws_eval_4shot_enout.py \