
python src/synthesis/de.py data_synthesis \
    --seed_dataset hotpotqa \
    --model gpt-4o \
    --seed_id 0

python src/synthesis/de.py data_synthesis \
    --seed_dataset eli5 \
    --model gpt-4o \
    --seed_id 0

python src/synthesis/de.py data_synthesis \
    --seed_dataset nq \
    --model gpt-4o \
    --seed_id 0

python src/synthesis/de.py data_synthesis \
    --seed_dataset fever \
    --model gpt-4o \
    --seed_id 0

python src/synthesis/de.py data_synthesis \
    --seed_dataset wow \
    --model gpt-4o \
    --seed_id 0

python src/synthesis/de.py data_synthesis \
    --seed_dataset trex \
    --model gpt-4o \
    --seed_id 0
