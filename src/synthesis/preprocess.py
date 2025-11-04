''' preparing for the datasets '''

from tqdm import tqdm
import os
import json
import requests
import argparse
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


random_seed_pool = [42, 17, 56, 29, 81, 2, 64, 88, 37, 12]

corpus = {
    'wikipedia_2019_08_01': 'http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json'
}

gti = ''

kilt_datasets = {
    'hotpotqa': {
        'train': 'http://dl.fbaipublicfiles.com/KILT/hotpotqa-train-kilt.jsonl',
        'test': 'http://dl.fbaipublicfiles.com/KILT/hotpotqa-dev-kilt.jsonl',
    }, # multi-hop qa
    'eli5': {
        'train': 'http://dl.fbaipublicfiles.com/KILT/eli5-train-kilt.jsonl',
        'test': 'http://dl.fbaipublicfiles.com/KILT/eli5-dev-kilt.jsonl',
    }, # long-form qa
    'nq': {
        'train': 'http://dl.fbaipublicfiles.com/KILT/nq-train-kilt.jsonl',
        'test': 'http://dl.fbaipublicfiles.com/KILT/nq-dev-kilt.jsonl',
    }, # single-hop qa
    'fever': {
        'train': 'http://dl.fbaipublicfiles.com/KILT/fever-train-kilt.jsonl',
        'test': 'http://dl.fbaipublicfiles.com/KILT/fever-dev-kilt.jsonl',
    }, # fact-checking
    'wow': {
        'train': 'http://dl.fbaipublicfiles.com/KILT/wow-train-kilt.jsonl',
        'test': 'http://dl.fbaipublicfiles.com/KILT/wow-dev-kilt.jsonl',
    }, # dialogue generation
    'trex': {
        'train': 'http://dl.fbaipublicfiles.com/KILT/trex-train-kilt.jsonl',
        'test': 'http://dl.fbaipublicfiles.com/KILT/trex-dev-kilt.jsonl',
    }, # slot filling
    'wned': {
        'test': 'http://dl.fbaipublicfiles.com/KILT/wned-dev-kilt.jsonl',
    }, # entity linking
    'zs-re': {
        'test': 'http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-dev-kilt.jsonl',
    } # relation extraction
}

beir_test = {
    'climate-fever': {},
    'scifact': {},
    'fiqa': {},
}


def download(url: str,
             filename: str):
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0)) # total size in bytes
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=url.split('/')[-1])
    with open(filename, 'wb') as file:
        for data in r.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()


def process_format_kilt(filename: str):
    data = []
    with open(filename, 'r') as file:
        lines = [json.loads(line) for line in file]
    
    for line in lines:
        try:
            new_data = {
                '_id': line['id'],
                'input': line['input'],
                'output': line['output'][0]['answer'], # collect the best reference output
            }
            data.append(new_data)
        except Exception as e:
            logging.warning(f'{e}\nThe data format of this row is incorrect. Skip this data.')
            continue
    
    with open(filename, 'w') as file:
        for line in data:
            file.write(json.dumps(line) + '\n')
    
    logging.info(f'number of valid data for file {filename} is {len(data)}.')


def collect_kilt_dataset(dataset_name: str,
                         base_folder: str = './data',
                         train: bool = True):
    folder = os.path.join(base_folder, dataset_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    if train:
        download(kilt_datasets[dataset_name]['train'],
                 os.path.join(folder, 'train.jsonl'))
    download(kilt_datasets[dataset_name]['test'],
             os.path.join(folder, 'test.jsonl'))
    
    # dict_keys([_id, input, output])
    if train:
        process_format_kilt(os.path.join(folder, 'train.jsonl'))
    process_format_kilt(os.path.join(folder, 'test.jsonl'))


def random_seed_collection(dataset_name: str,
                           base_folder: str = './data',
                           max_num: int = 1000,
                           n: int = 1):
    folder = os.path.join(base_folder, dataset_name)
    for i in range(n):
        random.seed(random_seed_pool[i])

        with open(os.path.join(folder, 'train.jsonl'), 'r') as file:
            lines = [json.loads(line) for line in file]
        if len(lines) < max_num:
            max_num = len(lines)
        
        sampled_data = random.sample(lines, max_num)

        with open(os.path.join(folder, f'train_seed_{i}.jsonl'), 'w') as file:
            for line in sampled_data:
                file.write(json.dumps(line) + '\n')
        
        logging.info(f'collect {max_num} seed data in train_seed_{i}.jsonl for {folder}.')


def random_test_data_collection(dataset_name: str,
                                base_folder: str = './data',
                                max_num: int = 1000,
                                n: int = 1):
    folder = os.path.join(base_folder, dataset_name)
    for i in range(n):
        random.seed(random_seed_pool[i])

        with open(os.path.join(folder, 'test.jsonl'), 'r') as file:
            lines = [json.loads(line) for line in file]
        if len(lines) < max_num:
            max_num = len(lines)

        sampled_data = random.sample(lines, max_num)

        with open(os.path.join(folder, f'test_{i}.jsonl'), 'w') as file:
            for line in sampled_data:
                file.write(json.dumps(line) + '\n')
        
        logging.info(f'collect {max_num} test data in test_{i}.jsonl for {folder}.')


def process_fiqa(base_folder: str = './data'):
    folder = os.path.join(base_folder, 'fiqa')
    with open(os.path.join(folder, 'test.json'), 'r') as file:
        lines = [json.loads(line) for line in file]
    
    data = []
    for idx, line in enumerate(lines):
        try:
            new_data = {
                '_id': str(idx),
                'input': line['question'],
                'output': line['answer'],
            }
            data.append(new_data)
        except Exception as e:
            logging.warning(f'{e}\nThe data format of this row is incorrect. Skip this data.')
            continue

    with open(os.path.join(folder, 'test.jsonl'), 'w') as file:
        for line in data:
            file.write(json.dumps(line) + '\n')
    
    logging.info(f"number of valid data for file {os.path.join(folder, 'test.jsonl')} is {len(data)}.")


def process_scifact(base_folder: str = './data'):
    ''' all data used as test split for the amount of data is small '''
    folder = os.path.join(base_folder, 'scifact')
    with open(os.path.join(folder, 'queries.jsonl'), 'r') as file:
        lines = [json.loads(line) for line in file]
    
    data = []
    for line in lines:
        try:
            new_data = {
                '_id': line['_id'],
                'input': line['text'],
                'output': line['metadata'][list(line['metadata'].keys())[0]][0]['label']
            }
            data.append(new_data)
        except Exception as e:
            logging.warning(f'{e}\nThe data format of this row is incorrect. Skip this data.')
            continue
    
    with open(os.path.join(folder, 'test.jsonl'), 'w') as file:
        for line in data:
            file.write(json.dumps(line) + '\n')
    
    logging.info(f"number of valid data for file {os.path.join(folder, 'test.jsonl')} is {len(data)}.")


def process_climate_fever(base_folder: str = './data'):
    ''' all data used as test split for the amount of data is small '''
    folder = os.path.join(base_folder, 'climate-fever')
    with open(os.path.join(folder, 'queries.jsonl'), 'r') as file:
        lines = [json.loads(line) for line in file]
    
    data = []
    for line in lines:
        try:
            new_data = {
                '_id': line['_id'],
                'input': line['text'],
                'output': line['metadata']['claim_label']
            }
            data.append(new_data)
        except Exception as e:
            logging.warning(f'{e}\nThe data format of this row is incorrect. Skip this data.')
            continue
    
    with open(os.path.join(folder, 'test.jsonl'), 'w') as file:
        for line in data:
            file.write(json.dumps(line) + '\n')
    
    logging.info(f"number of valid data for file {os.path.join(folder, 'test.jsonl')} is {len(data)}.")


def split_100_words(text, max_words=100):
    words = text.split()
    
    substrings = []
    current_substring = []
    
    for word in words:
        current_substring.append(word)
        if len(current_substring) == max_words:
            substrings.append(' '.join(current_substring))
            current_substring = []
    
    if current_substring:
        substrings.append(' '.join(current_substring))
    
    return substrings


def split_corpus_to_100(in_corpus_file='./wikipedia_2019_08_01.jsonl',
                        out_corpus_file='./wikipedia_100_2019_08_01.jsonl',
                        chunk_size=1000):
    ''' split articles into passages with max length of 100 words, each passage also has its wikipedia title '''
    with open(in_corpus_file, 'r') as infile, open(out_corpus_file, 'w') as outfile:
        chunk = []
        idx = 0
        for line in infile:
            obj = json.loads(line)
            passages = split_100_words(obj['text'])
            for i, passage in enumerate(passages):
                new_obj = {
                    '_id': obj['_id'] + str(i),
                    'title': obj['title'],
                    'text': passage,
                }
                chunk.append(new_obj)
                idx += 1
            
                if (idx + 1) % chunk_size == 0:
                    print('1000 finished.')
                    for item in chunk:
                        outfile.write(json.dumps(item) + '\n')
                    chunk = []
                    idx = 0
        
        if chunk:
            for item in chunk:
                outfile.write(json.dumps(item) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to collect and process.')
    parser.add_argument('--base', type=str, default='./data', help='The base folder.')
    args = parser.parse_args()
    if args.dataset in ['hotpotqa', 'eli5', 'nq', 'fever', 'wow', 'trex']:
        collect_kilt_dataset(args.dataset, args.base, train=True)
        random_seed_collection(args.dataset, args.base, max_num=1000, n=5)
        random_test_data_collection(args.dataset, args.base, max_num=1000, n=1)
    elif args.dataset in ['wned', 'triviaqa', 'zs-re']:
        collect_kilt_dataset(args.dataset, args.base, train=False)
        random_test_data_collection(args.dataset, args.base, max_num=1000, n=1)
    elif args.dataset in ['wikipedia']:
        pass
    elif args.dataset == 'fiqa':
        process_fiqa(args.base)
        random_test_data_collection(args.dataset, args.base, max_num=1000, n=1)
    elif args.dataset == 'scifact':
        process_scifact(args.base)
        random_test_data_collection(args.dataset, args.base, max_num=1000, n=1)
    elif args.dataset == 'climate-fever':
        process_climate_fever(args.base)
        random_test_data_collection(args.dataset, args.base, max_num=1000, n=1)
    else:
        logging.error(f'invalid dataset name: {args.dataset}.')


if __name__ == '__main__':
    main()
