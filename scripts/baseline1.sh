export CUDA_VISIBLE_DEVICES=0

######################Retrieval Part##############################

echo "Starting script execution..."

# corpus_path_list=("data/wikipedia_100_2019_08_01.jsonl" "data/climate-fever/corpus.jsonl" "data/fiqa/corpus.jsonl" "data/scifact/corpus.jsonl")
corpus_path_list=("data/pooled_corpus.jsonl")
retriever_list=("contriever" "bge-base" "aar_contriever")

for retriever in "${retriever_list[@]}"; do
    echo "Processing retriever: $retriever"

    for corpus_path in "${corpus_path_list[@]}"; do

        python -m src.inference.rag \
            --retriever "$retriever" \
            --corpus_path "$corpus_path" \
            --topk 10

    done

echo "$retriever finished."
done

echo "All finished.."
###########################################################
