export CUDA_VISIBLE_DEVICES=1

generator="../hf/qwen2.5-3b-instruct"
generator_short="qwen2.5"

######################Generation Part(No Retrieval)(Qwen-2.5)##############################
datasets=("nq" "hotpotqa" "fever" "eli5" "wow" "trex" "zs-re" "climate-fever" "scifact" "fiqa")

for dataset in "${datasets[@]}"; do

    data_path="data/${dataset}/test_0.jsonl"
    save_folder="data/${dataset}/outputs_test_0_no_retrieval_${generator_short}.jsonl"

    python -m src.inference.downstream \
        --dataset "$dataset" \
        --data "$data_path" \
        --generator "$generator" \
        --save_path "$save_folder"

done
##########################################################################


######################Generation Part(with Retrieval)(Qwen-2.5)##############################
datasets=("nq" "hotpotqa" "fever" "eli5" "wow" "trex" "zs-re" "climate-fever" "scifact" "fiqa")
retriever_list=("contriever" "bge-base" "aar_contriever")

for retriever in "${retriever_list[@]}"; do

    for dataset in "${datasets[@]}"; do

        data_path="data/${dataset}/test_0_w_passages_${retriever}.jsonl"
        save_folder="data/${dataset}/outputs_test_0_${retriever}_${generator_short}.jsonl"

        python -m src.inference.downstream \
            --retrieval \
            --dataset "$dataset" \
            --data "$data_path" \
            --generator "$generator" \
            --topk 3 \
            --save_path "$save_folder"

    done

done
##########################################################################
