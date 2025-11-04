export CUDA_VISIBLE_DEVICES=0

# LLaMA-3-8b
# python -m src.attribution.attribute \
#     --model_name llama-3-8b
#     --model ../hf/llama-3-8b-instruct \
#     --method perturbation_based \
#     --data data/train/0/all_filtered.jsonl \
#     --seed_id 0

# Qwen-2.5-3b
python -m src.attribution.attribute \
    --model_name qwen2.5-3b \
    --model ../hf/qwen2.5-3b-instruct \
    --method perturbation_based \
    --data data/train/0/all_filtered.jsonl \
    --seed_id 0

