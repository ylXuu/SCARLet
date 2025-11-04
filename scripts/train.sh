export CUDA_VISIBLE_DEVICES=0

python -m src.training.train \
    --data data/train/0/train.jsonl \
    --model_path ../hf/contriever \
    --strategy lists \
    --ckpt_path ckpt/contriever \
    --save_path ckpt/contriever/models

