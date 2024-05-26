random_port=$((RANDOM % 10000 + 20000))

echo "Using random port: $random_port"

export CUDA_VISIBLE_DEVICES=3

export HF_HOME=/home/nfs03/zhangsm/multiL-transfer-interpretability/llm-latent-language/data

torchrun --nproc_per_node 1 --master_port $random_port ../zsm_emotion_2shot.py \
