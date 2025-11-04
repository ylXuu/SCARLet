''' label each passage of each data with a utility(attribution) score '''

from src.attribution.context_attributor import ContextAttributor
from src.synthesis.prompts import TASK_POOL
from tqdm import tqdm
import os
import json
import torch
import argparse
import logging


logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
PROMPT_TEMPLATE = '{task_instruction}\n\nPassages:\n{context}\n\nInput: {input}'


def perturbation_based(model_name: str,
                       model_path: str,
                       data_file: str,
                       base_folder: str = './data',
                       seed_id: str = '0',
                       num_ablations: int = 64,):
    
    logging.info('loading attributor...')
    cc = ContextAttributor.from_pretrained(
        model_path,
        model_kwargs={'device_map': 'auto', 'torch_dtype': torch.float16},
        tokenizer_kwargs={'padding_side': 'left'},
        num_ablations=num_ablations,
        prompt_template=PROMPT_TEMPLATE,
    )

    with open(data_file, 'r') as file:
        all_data = [json.loads(line) for line in file]
    
    # classify every task
    seed_dataset_list = {}
    for data in all_data:
        seed_dataset = data['seed_dataset']
        if seed_dataset not in seed_dataset_list:
            seed_dataset_list[seed_dataset] = []
        seed_dataset_list[seed_dataset].append(data)
    
    if len(seed_dataset_list.keys()) != len(TASK_POOL.keys()):
        logging.error('Wrong dataset!') # BUG: now the lengths are not equal
    
    annotated_data = []
    for seed_dataset in seed_dataset_list.keys():
        logging.info(f'Attributing data of seed dataset {seed_dataset}.')
        for data in tqdm(seed_dataset_list[seed_dataset], desc=f'{seed_dataset}'):
            if data['input'] is None or len(data['input']) == 0 or data['output'] is None or len(data['output']) == 0:
                # incomplete data
                continue

            cc.init_query_context(
                query=data['input'],
                passages=['Title: ' + doc['title'] + '\n' + 'Text: ' + doc['text'] for doc in data['context']],
                task_instruction=TASK_POOL[data['seed_dataset']]['task_instruction'],
                ground_truth_answer=data['output']
            )
            results, _ = cc.get_attributions()
            passages = data['context']
            for idx, passage in enumerate(passages):
                passage['utility'] = results[idx] # dict_keys(['title', 'text', 'score', 'utility'])
            
            annotated_data.append({
                '_id': data['_id'],
                'task_name': data['task_name'],
                'seed_dataset': data['seed_dataset'],
                'input': data['input'],
                'context': passages,
                'output': data['output'],
            })
    
    output_filename = os.path.join(base_folder, 'train', seed_id, f'all_annotated_{model_name}.jsonl')
    with open(output_filename, 'w') as file:
        for data in annotated_data:
            file.write(json.dumps(data) + '\n')
    
    return {
        'output_file': output_filename
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--model', required=True, type=str, help='The attributor model name or path.')
    parser.add_argument('--method', required=True, type=str, help='The attribution method.')
    parser.add_argument('--data', required=True, type=str, help='The data file path.')
    parser.add_argument('--seed_id', type=str, required=True)
    parser.add_argument('--base', type=str, default='./data', help='The base folder.')
    args = parser.parse_args()

    if args.method == 'perturbation_based':
        output = perturbation_based(args.model_name, args.model, args.data, args.base, args.seed_id)
        logging.info(f"Done.\nOutput file is stored in {output['output_file']}.")


if __name__ == '__main__':
    main()
