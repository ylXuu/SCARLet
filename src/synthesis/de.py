''' overall process of data enhancement '''

from synthesize import (
    get_prompt_for_new_data_one_step,
    parse_teacher_model_output,
    data_checking_filetering,
)
from wikidata import parse_entities, collect_entities
from retrieval import (
    CorpusLoader,
    QueriesLoader,
    DenseRetrievalExactSearch as DR,
    DenseRetrievalExactSearchMultiDatasets as DRMD,
)
from retrievers import DPR, BGE
from prompts import TASK_POOL
from typing import List, Dict, Any
from tqdm import tqdm
import os
import json
import openai
import argparse
import logging


logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def entities_extraction(dataset: str,
                        base_folder: str = './data',
                        seed_id: str = '0'):
    '''
    output entities for each seed data
    '''

    filename = os.path.join(base_folder, dataset, f'train_seed_{seed_id}.jsonl')
    with open(filename, 'r') as file:
        # dict_keys(['_id', 'input', 'output'])
        seed_dataset = [json.loads(line) for line in file]
    
    # dict_keys(['_id', 'entities'])
    output_seed_dataset = []
    for seed_data in tqdm(seed_dataset, desc=f'entities extraction from {dataset}'):
        entities = []
        # extract from input & output
        entities.extend(parse_entities(text=seed_data['input']))
        entities.extend(parse_entities(text=seed_data['output']))
        if entities:
            output_seed_dataset.append({
                '_id': seed_data['_id'],
                # 'input': seed_data['input'],
                # 'reference_output': seed_data['output'],
                'entities': entities
            })
            # else, discard
    
    output_filename = os.path.join(base_folder, dataset, f'train_seed_{seed_id}_entities.jsonl')
    with open(output_filename, 'w') as file:
        for line in output_seed_dataset:
            file.write(json.dumps(line) + '\n')
    
    return {
        'output_file': output_filename,
        'num_data': len(output_seed_dataset),
        'discard_rate': 100 * (len(seed_dataset) - len(output_seed_dataset)) / len(seed_dataset),
        'avg_num_entities': sum([len(seed_data['entities']) for seed_data in output_seed_dataset]) / len(output_seed_dataset)
    }


def entities_retrieval(dataset: str,
                       base_folder: str = './data',
                       seed_id: str = '0'):
    '''
    retrieve related entities for each seed data
    '''

    filename = os.path.join(base_folder, dataset, f'train_seed_{seed_id}_entities.jsonl')
    with open(filename, 'r') as file:
         # dict_keys(['_id', 'entities'])
        seed_dataset = [json.loads(line) for line in file]
    
    num_unexpanded = 0
    num_new_entities = 0
    for seed_data in tqdm(seed_dataset):
        entities = seed_data['entities']
        seed_data['entities'] = collect_entities(entities)
        if len(seed_data['entities']) == len(entities):
            # not expanded
            num_unexpanded += 1
        else:
            # expanded
            num_new_entities += (len(seed_data['entities']) - len(entities))
    
    with open(filename, 'w') as file:
        for line in seed_dataset:
            file.write(json.dumps(line) + '\n')
    
    return {
        'output_file': filename,
        'expand_rate': 100 * (len(seed_dataset) - num_unexpanded) / len(seed_dataset),
        'avg_num_new_entities': num_new_entities / (len(seed_dataset) - num_unexpanded), # average number of new entities expanded
        'avg_num_entities': sum([len(seed_data['entities']) for seed_data in seed_dataset]) / len(seed_dataset)
    } # num of data unchanges


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
    filename_list = [os.path.join(base_folder, dataset, f'train_seed_{seed_id}_entities.jsonl') for dataset in dataset_list]
    corpus = CorpusLoader(corpus_path=corpus_file).load()
    queries_list, mapping_list = [], []
    for filename in filename_list:
        queries, mapping = QueriesLoader(data_path=filename, query_type='entity').load()
        queries_list.append(queries)
        mapping_list.append(mapping)
    
    if model_name == 'dpr':
        model = DPR((
            'facebook/dpr-question_encoder-multiset-base',
            'facebook/dpr-ctx_encoder-multiset-base'
        ))
        retriever = DRMD(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bge-base':
        model = BGE('../hf/bge-base-en-v1.5')
        retriever = DRMD(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bge-large':
        model = BGE('../hf/bge-large-en-v1.5')
        retriever = DRMD(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bm25':
        pass

    results_list = retriever.search(corpus, queries_list, top_k=top_k, score_function="cos_sim", return_sorted=False)

    output_filename_list = [os.path.join(base_folder, dataset, f'train_seed_{seed_id}_passages.jsonl') for dataset in dataset_list]
    for output_filename, results, mapping in zip(output_filename_list, results_list, mapping_list):
        with open(output_filename, 'w') as file:
            for query_id, sub_query_id_list in mapping.items():
                # merge sub queries' results
                ranking_scores = {}
                for sub_query_id in sub_query_id_list:
                    ranking_scores.update(results[sub_query_id])
                
                sorted_scores = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
                docs = [{'title': corpus[doc_id].get('title'), 'text': corpus[doc_id].get('text'), 'score': score} for doc_id, score in sorted_scores[:top_k]]
                line = {'_id': query_id, 'docs': docs}
                file.write(json.dumps(line) + '\n')
    
    return {
        'output_file_list': output_filename_list
    }


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
    filename = os.path.join(base_folder, dataset, f'train_seed_{seed_id}_entities.jsonl')
    queries, mapping = QueriesLoader(data_path=filename, query_type='entity').load()
    corpus = CorpusLoader(corpus_path=corpus_file).load()

    if model_name == 'dpr':
        model = DPR((
            'facebook/dpr-question_encoder-multiset-base',
            'facebook/dpr-ctx_encoder-multiset-base'
        )),
        retriever = DR(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bge-base':
        model = BGE('../hf/bge-base-en-v1.5')
        retriever = DR(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bge-large':
        model = BGE('../hf/bge-large-en-v1.5')
        retriever = DR(model=model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    elif model_name == 'bm25':
        pass

    results = retriever.search(corpus, queries, top_k=top_k, score_function='cos_sim', return_sorted=False)

    # dict_keys(['_id', 'docs'])
    output_filename = os.path.join(base_folder, dataset, f'train_seed_{seed_id}_passages.jsonl')    
    with open(output_filename, 'w') as file:
        for query_id, sub_query_id_list in mapping.items():
            # merge sub queries' results
            ranking_scores = {}
            for sub_query_id in sub_query_id_list:
                ranking_scores.update(results[sub_query_id])
            
            sorted_scores = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
            docs = [{'title': corpus[doc_id].get('title'), 'text': corpus[doc_id].get('text'), 'score': score} for doc_id, score in sorted_scores[:top_k]]
            line = {'_id': query_id, 'docs': docs}
            file.write(json.dumps(line) + '\n')

    return {
        'output_file': output_filename
    }

def data_synthesis(seed_dataset: str,
                   model_name: str,
                   task_list: Dict[str, Any], # TASK_POOL
                   base_folder: str = './data',
                   seed_id: str = '0',):
    '''
    synthesize new data for different tasks under the shared context
    '''
    passages_filename = os.path.join(base_folder, seed_dataset, f'train_seed_{seed_id}_passages.jsonl')
    seed_filename = os.path.join(base_folder, seed_dataset, f'train_seed_{seed_id}.jsonl')
    with open(seed_filename, 'r') as file:
        seed_data = [json.loads(line) for line in file] # dict_keys(['_id', 'input', 'output'])
    with open(passages_filename, 'r') as file:
        passages_data = [json.loads(line) for line in file] # dict_keys(['_id', 'docs'])

    OPEN_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPEN_API_URL = os.environ.get('OPEN_API_URL')
    client = openai.OpenAI(
        base_url=OPEN_API_URL,
        api_key=OPEN_API_KEY
    )
    
    new_dataset = [] # dict_keys(['_id', 'task_name', 'seed_dataset', 'input', 'context', 'output'])
    for idx, data in enumerate(seed_data):
        logging.info(f'seed data - {idx + 1}/{len(seed_data)}')

        # match input & passages by _id
        passages = next((item for item in passages_data if item['_id'] == data['_id']), None)
        if passages is None:
            logging.warning('The \'_id\' field of the paragraph and the original data does not match')
            continue
        passages = passages['docs']

        for task in task_list.keys():
            logging.info(f'synthesizing for task - {task}')
            output = get_prompt_for_new_data_one_step(
                context='\n\n'.join([passage['text'] for passage in passages]),
                task_name=task_list[task]['task_name'],
                task_description=task_list[task]['task_description'],
                task_example_input=task_list[task]['task_examples'][0]['input'],
                task_example_output=task_list[task]['task_examples'][0]['output'],
                client=client,
                model_name=model_name
            )
            if output is None:
                continue
            new_data = parse_teacher_model_output(output)
            if new_data is None:
                continue
            new_dataset.append({
                '_id': seed_dataset + '_' + task + '_' + str(idx),
                'task_name': task_list[task]['task_name'],
                'seed_dataset': seed_dataset,
                'input': new_data[0],
                'output': new_data[1],
                'context': passages,
            })

    if not os.path.exists(os.path.join(base_folder, 'train', seed_id)):
        os.makedirs(os.path.join(base_folder, 'train', seed_id))
    output_filename = os.path.join(base_folder, 'train', seed_id, f'{seed_dataset}.jsonl')
    with open(output_filename, 'w') as file:
        for new_data in new_dataset:
            file.write(json.dumps(new_data) + '\n')

    return {
        'output_file': output_filename,
        'num_data': len(new_dataset),
        'loss_rate': 100 * (len(seed_data) * len(task_list.keys()) - len(new_dataset)) / (len(seed_data) * len(task_list.keys()))
    }


def data_filtering(model_name: str,
                   task_list: Dict[str, Any], # TASK_POOL
                   base_folder: str = './data',
                   seed_id: str = '0'):
    ''' Instruct the teacher model to check and filter the synthesized data '''
    train_file_list = [os.path.join(base_folder, 'train', seed_id, f'{dataset}.jsonl') for dataset in task_list.keys()]
    new_dataset = []
    for train_file in train_file_list:
        with open(train_file, 'r') as file:
            new_dataset.extend([json.loads(line) for line in file])

    OPEN_API_KEY = os.environ.get('OPENAI_API_KEY')
    OPEN_API_URL = os.environ.get('OPEN_API_URL')
    client = openai.OpenAI(
        base_url=OPEN_API_URL,
        api_key=OPEN_API_KEY
    )
    
    filtered_new_dataset = []
    for new_data in tqdm(new_dataset):
        output = data_checking_filetering(
            context='\n\n'.join([passage['text'] for passage in new_data['context']]),
            task_name=new_data['task_name'],
            task_description=task_list[new_data['seed_dataset']]['task_description'],
            task_example_input=task_list[new_data['seed_dataset']]['task_examples'][0]['input'],
            task_example_output=task_list[new_data['seed_dataset']]['task_examples'][0]['output'],
            client=client,
            input=new_data['input'],
            output=new_data['output'],
            model_name=model_name
        )
        if output:
            filtered_new_dataset.append(new_data)
    
    output_filename = os.path.join(base_folder, 'train', seed_id, 'all_filtered.jsonl')
    with open(output_filename, 'w') as file:
        for new_data in filtered_new_dataset:
            file.write(json.dumps(new_data) + '\n')
    
    return {
        'output_file': output_filename,
        'initial_num_new_data': len(new_dataset),
        'final_num_new_data': len(filtered_new_dataset),
        'loss_rate': 100 * (len(new_dataset) - len(filtered_new_dataset)) / len(new_dataset)
    }



def main():
    parser = argparse.ArgumentParser(description='Main parser for different periods of DE.')
    subparsers = parser.add_subparsers(dest='command')

    parser_step_1 = subparsers.add_parser('entities_extraction')
    parser_step_1.add_argument('--seed_dataset', required=True, type=str, help='The preprocessed dataset of seed data.')
    parser_step_1.add_argument('--seed_id', type=str, required=True)
    parser_step_1.add_argument('--base', type=str, default='./data', help='The base folder.')

    parser_step_2 = subparsers.add_parser('entities_retrieval')
    parser_step_2.add_argument('--seed_dataset', required=True, type=str, help='The preprocessed dataset of seed data.')
    parser_step_2.add_argument('--seed_id', type=str, required=True)
    parser_step_2.add_argument('--base', type=str, default='./data', help='The base folder.')

    parser_step_3 = subparsers.add_parser('passages_retrieval')
    parser_step_3.add_argument('--seed_dataset', required=True, type=str, help='The preprocessed dataset of seed data.')
    parser_step_3.add_argument('--seed_id', type=str, required=True)
    parser_step_3.add_argument('--base', type=str, default='./data', help='The base folder.')
    parser_step_3.add_argument('--corpus', required=True, type=str, help='The corpus file path.')
    parser_step_3.add_argument('--retriever', required=True, type=str, help='The retriever model path.')
    parser_step_3.add_argument('--top_k', default=3, type=int, help='Each retrieval retains top-k passages with the highest scores.')

    parser_step_4 = subparsers.add_parser('data_synthesis')
    parser_step_4.add_argument('--seed_dataset', required=True, type=str, help='The preprocessed dataset of seed data.')
    parser_step_4.add_argument('--model', type=str, default='gpt-4o', help='The name of the teacher model.') # gpt-4o-2024-08-06
    parser_step_4.add_argument('--seed_id', type=str, required=True)
    parser_step_4.add_argument('--base', type=str, default='./data', help='The base folder.')

    parser_step_5 = subparsers.add_parser('data_filtering')
    parser_step_5.add_argument('--seed_id', type=str, required=True)
    parser_step_5.add_argument('--base', type=str, default='./data', help='The base folder.')
    parser_step_5.add_argument('--model', type=str, default='gpt-4o', help='The name of the teacher model.')

    args = parser.parse_args()

    if args.command == 'entities_extraction':
        logging.info(f'Step 1 - entities extraction\nDataset: {args.seed_dataset}\n')
        step_1 = entities_extraction(args.seed_dataset, args.base, args.seed_id)
        logging.info(f"Done.\nOutput file is stored in {step_1['output_file']}\n"
                 f"Numebr of valid data: {step_1['num_data']}\n"
                 f"Discard Rate: {step_1['discard_rate']}\n"
                 f"Averaged Number of Entities: {step_1['avg_num_entities']}")
    
    elif args.command == 'entities_retrieval':
        logging.info(f'Step 2 - entities retrieval\nDataset: {args.seed_dataset}\n')
        step_2 = entities_retrieval(args.seed_dataset, args.base, args.seed_id)
        logging.info(f"Done.\nOutput file is stored in {step_2['output_file']}\n"
                     f"Expand Rate: {step_2['expand_rate']}\n"
                     f"Average Number of New Entities Expanded: {step_2['avg_num_new_entities']}\n"
                     f"Averaged Number of Entities: {step_2['avg_num_entities']}")
    
    elif args.command == 'passages_retrieval':
        logging.info(f'Step 3 - passages retrieval\nDataset: {args.seed_dataset}')
        if args.seed_dataset == 'task_pool':
            step_3 = passages_retrieval_multiset(TASK_POOL.keys(), args.corpus, args.base, args.seed_id,
                                    model_name=args.retriever, top_k=args.top_k)
            logging.info(f"Done.\nOutput files are stored in {step_3['output_file_list']}.")
        else:
            step_3 = passages_retrieval(args.seed_dataset, args.corpus, args.base, args.seed_id,
                                    model_name=args.retriever, top_k=args.top_k)
            logging.info(f"Done.\nOutput file is stored in {step_3['output_file']}.")

    elif args.command == 'data_synthesis':
        logging.info(f'Step 4 - data synthesis\nSeed dataset: {args.seed_dataset}')
        step_4 = data_synthesis(args.seed_dataset, args.model, TASK_POOL, args.base, args.seed_id)
        logging.info(f"Done.\nOutput file is stored in {step_4['output_file']}\n"
                     f"The number of new data is {step_4['num_data']}\n"
                     f"The loss rate of the generated data is {step_4['loss_rate']}")
    
    elif args.command == 'data_filtering':
        logging.info(f'Step 5 - data filtering\nSeed: {args.seed_id}')
        step_5 = data_filtering(args.model, TASK_POOL, args.base, args.seed_id)
        logging.info(f"Done.\nOutput file is stored in {step_5['output_file']}\n"
                     f"The initial number of new data is {step_5['initial_num_new_data']}\n"
                     f"The final number of new data is {step_5['final_num_new_data']}\n"
                     f"The loss rate of filtering process is {step_5['loss_rate']}")
    
    else:
        logging.error(f'Invalid command: {args.command}')



    

if __name__ == '__main__':
    main()







