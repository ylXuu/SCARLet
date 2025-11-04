export CUDA_VISIBLE_DEVICES=0

python src/synthesis/de.py passages_retrieval \
    --seed_dataset task_pool \
    --seed_id 0 \
    --corpus data/wikipedia_100_2019_08_01.jsonl \
    --retriever bge-base \
    --top_k 10

python src/synthesis/de.py passages_retrieval \
    --seed_dataset hotpotqa \
    --seed_id 0 \
    --corpus data/wikipedia_100_2019_08_01.jsonl \
    --retriever bge-base \
    --top_k 10

python src/synthesis/de.py passages_retrieval \
    --seed_dataset eli5 \
    --seed_id 0 \
    --corpus data/wikipedia_100_2019_08_01.jsonl \
    --retriever bge-base \
    --top_k 10

python src/synthesis/de.py passages_retrieval \
    --seed_dataset nq \
    --seed_id 0 \
    --corpus data/wikipedia_100_2019_08_01.jsonl \
    --retriever bge-base \
    --top_k 10

python src/synthesis/de.py passages_retrieval \
    --seed_dataset fever \
    --seed_id 0 \
    --corpus data/wikipedia_100_2019_08_01.jsonl \
    --retriever bge-base \
    --top_k 10

python src/synthesis/de.py passages_retrieval \
    --seed_dataset wow \
    --seed_id 0 \
    --corpus data/wikipedia_100_2019_08_01.jsonl \
    --retriever bge-base \
    --top_k 10

python src/synthesis/de.py passages_retrieval \
    --seed_dataset trex \
    --seed_id 0 \
    --corpus data/wikipedia_100_2019_08_01.jsonl \
    --retriever bge-base \
    --top_k 10
