''' retrieval of vanilla rag baseline '''


from src.synthesis.prompts import TASK_POOL
from src.synthesis.retrieval import (
    CorpusLoader,
    QueriesLoader,
    DenseRetrievalExactSearch as DR,
    DenseRetrievalExactSearchMultiDatasets as DRMD,
)
from src.synthesis.retrievers import (
    BGE,
    DPR,
    Contriever,
)
from typing import List
import os
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def passages_retrieval(dataset: str,
                       corpus_file: str,
                       base_folder: str = './data',
                       seed_id: str = '0',
                       model_name: str = 'dpr',
                       top_k: int = 5,
                       batch_size: int = 128,
                       corpus_chunk_size: int = 100000):
    '''
    retrieve relevant passages for shared context construction based on entities
    '''
    filename = os.path.join(base_folder, dataset, f'test_{seed_id}.jsonl')
    queries, inputs, outputs = QueriesLoader(
        data_path=filename, query_type='input', task_instruction=TASK_POOL[dataset]['retrieval_instruction']).load()
    corpus = CorpusLoader(corpus_path=corpus_file).load()

    if model_name == 'dpr':
        model = DPR((
            'facebook/dpr-question_encoder-multiset-base',
            'facebook/dpr-ctx_encoder-multiset-base'
        )),
        retriever = DR(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bge-base' or model_name == 'bge':
        model = BGE('../hf/bge-base-en-v1.5')
        retriever = DR(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bge-large':
        model = BGE('../hf/bge-large-en-v1.5')
        retriever = DR(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'contriever':
        model = Contriever('../hf/contriever')
        retriever = DR(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'replug_contriever':
        pass
    elif model_name == 'aar_contriever':
        model = Contriever('../hf/aar-contriever')
        retriever = DR(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'scarlet_bge':
        model = BGE('./ckpt/bge-base/checkpoint-10000')
        retriever = DR(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    else:
        logging.error(f'Wrong retriever: {model_name}')

    results = retriever.search(corpus, queries, top_k=top_k, score_function='cos_sim', return_sorted=False)

    # dict_keys(['_id', 'input', 'docs', 'output'])
    output_filename = os.path.join(base_folder, dataset, f'test_{seed_id}_w_passages_{model_name}.jsonl')    
    with open(output_filename, 'w') as file:
        for query_id, ranking_scores in results.items():
            sorted_scores = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
            docs = [{'title': corpus[doc_id].get('title'), 'text': corpus[doc_id].get('text'), 'score': score} for doc_id, score in sorted_scores]
            line = {'_id': query_id, 'input': inputs[query_id], 'docs': docs, 'output': outputs[query_id]}
            file.write(json.dumps(line) + '\n')

    return {
        'output_file': output_filename
    }


def passages_retrieval_multiset(dataset_list: List[str],
                                corpus_file: str,
                                base_folder: str = './data',
                                seed_id: str = '0',
                                model_name: str = 'dpr',
                                top_k: int = 5,
                                batch_size: int = 128,
                                corpus_chunk_size: int = 10000):
    '''
    retrieve relevant passages for shared context construction based on entities, 
    for multiple datasets at the same time
    '''
    corpus = CorpusLoader(corpus_path=corpus_file).load()
    queries_list, inputs_list, outputs_list = [], [], []
    for dataset in dataset_list:
        filename = os.path.join(base_folder, dataset, f'test_{seed_id}.jsonl')
        queries, inputs, outputs = QueriesLoader(
            data_path=filename, query_type='input', task_instruction=TASK_POOL[dataset]['retrieval_instruction']).load()
        queries_list.append(queries)
        inputs_list.append(inputs)
        outputs_list.append(outputs)
    
    if model_name == 'dpr':
        model = DPR((
            'facebook/dpr-question_encoder-multiset-base',
            'facebook/dpr-ctx_encoder-multiset-base'
        )),
        retriever = DRMD(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bge-base' or model_name == 'bge':
        model = BGE('../hf/bge-base-en-v1.5')
        retriever = DRMD(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bge-large':
        model = BGE('../hf/bge-large-en-v1.5')
        retriever = DRMD(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'contriever':
        model = Contriever('../hf/contriever')
        retriever = DRMD(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'replug_contriever':
        pass
    elif model_name == 'aar_contriever':
        model = Contriever('../hf/aar-contriever')
        retriever = DRMD(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'scarlet_bge':
        model = BGE('./ckpt/bge-base/checkpoint-10000')
        retriever = DRMD(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    else:
        logging.error(f'Wrong retriever: {model_name}')

    results_list = retriever.search(corpus, queries_list, top_k=top_k, score_function="cos_sim", return_sorted=False)

    output_filename_list = [os.path.join(base_folder, dataset, f'test_{seed_id}_w_passages_{model_name}_pooled.jsonl') for dataset in dataset_list]
    for output_filename, results, inputs, outputs in zip(output_filename_list, results_list, inputs_list, outputs_list):
        # dict_keys(['_id', 'input', 'docs', 'output'])
        with open(output_filename, 'w') as file:
            for query_id, ranking_scores in results.items():
                sorted_scores = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
                docs = [{'title': corpus[doc_id].get('title'), 'text': corpus[doc_id].get('text'), 'score': score} for doc_id, score in sorted_scores]
                line = {'_id': query_id, 'input': inputs[query_id], 'docs': docs, 'output': outputs[query_id]}
                file.write(json.dumps(line) + '\n')
    
    return {
        'output_file_list': output_filename_list
    }


def main():
    wiki_datasets = ['nq', 'hotpotqa', 'eli5', 'fever', 'wow', 'trex', 'zs-re']
    beir_datasets = ['fiqa', 'climate-fever', 'scifact']
    parser = argparse.ArgumentParser()
    parser.add_argument('--retriever', required=True, type=str, help='The retriever model path.')
    parser.add_argument('--corpus_path', required=True, type=str, help='The corpus file path.')
    parser.add_argument('--topk', default=5, required=True, type=int)
    args = parser.parse_args()

    if 'wiki' in args.corpus_path:
        outputs = passages_retrieval_multiset(
            dataset_list=wiki_datasets,
            corpus_file=args.corpus_path,
            model_name=args.retriever,
            top_k=args.topk
        )
        logging.info(outputs)
    elif 'fiqa' in args.corpus_path:
        outputs = passages_retrieval(
            dataset='fiqa',
            corpus_file=args.corpus_path,
            model_name=args.retriever,
            top_k=args.topk
        )
        logging.info(outputs)
    elif 'climate' in args.corpus_path:
        outputs = passages_retrieval(
            dataset='climate-fever',
            corpus_file=args.corpus_path,
            model_name=args.retriever,
            top_k=args.topk
        )
        logging.info(outputs)
    elif 'scifact' in args.corpus_path:
        outputs = passages_retrieval(
            dataset='scifact',
            corpus_file=args.corpus_path,
            model_name=args.retriever,
            top_k=args.topk
        )
        logging.info(outputs)
    elif 'pooled' in args.corpus_path:
        outputs = passages_retrieval_multiset(
            dataset_list=wiki_datasets + beir_datasets,
            corpus_file=args.corpus_path,
            model_name=args.retriever,
            top_k=args.topk
        )


if __name__ == '__main__':
    main()
