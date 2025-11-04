''' sample positive and negative training samples according to utility scores '''

from tqdm import tqdm
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from src.synthesis.prompts import TASK_POOL
import os
import json
import argparse
import warnings
import logging
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


# Different sampling strategies:
# 1) One-dimensional clustering sampling.
# 2) One-dimensional clustering sampling, using only the highest-scoring document and the set of low-scoring documents obtained from clustering.
# 3) The highest-scoring document is a positive sample, and the lowest-scoring document is a negative sample (this seems unreasonable at present, as there is usually more than one negative sample).
##### Sampling only needs to be done once to obtain a list of positive samples and a list of negative samples, and then the three cases can be extracted from them respectively.


def sampling_1D(passages: List[Dict],
                n_clusters: int = 3,
                seed: int = 42):
    # dict_keys(['title', 'text', 'score', 'utility'])
    scores = [passage['utility'] for passage in passages]
    scores = np.array(scores).reshape(-1, 1)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        kmeans.fit(scores)

        for warning in w:
            if warning.category == ConvergenceWarning:
                logging.warning(f'processing ConvergenceWarning: {scores}')
                return [], []
    labels = kmeans.labels_

    # it is uncertain which label corresponds to the positive region or the negative region
    regions = [
        [score[0] for i, score in enumerate(scores) if labels[i] == idx] for idx in range(n_clusters)
    ]
    if False in [len(region) != 0 for region in regions]:
        return [], []
    top_scores = [max(region) for region in regions]
    idx_max = np.argmax(top_scores) # label of positive region
    idx_min = np.argmin(top_scores) # label of negative region

    poss = []
    negs = []
    for i in range(len(passages)):
        if labels[i] == idx_max:
            poss.append(passages[i])
        elif labels[i] == idx_min:
            negs.append(passages[i])
    
    return poss, negs

# scores = [10.9, 10.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str, help='The data file path.')
    parser.add_argument('--generator_name', type=str, required=True)
    parser.add_argument('--seed_id', type=str, required=True)
    parser.add_argument('--base', type=str, default='./data', help='The base folder.')
    args = parser.parse_args()

    with open(args.data, 'r') as file:
        # dict_keys(['_id', 'task_name', 'seed_dataset', 'input', 'context', 'output'])
        data = [json.loads(line) for line in file]
    
    training_data = []
    for instance in tqdm(data):
        poss, negs = sampling_1D(instance['context'])
        if len(poss) == 0 or len(negs) == 0 or len(poss) >= len(negs):
            continue
        training_data.append({
            '_id': instance['_id'],
            'task_name': instance['task_name'],
            'seed_dataset': instance['seed_dataset'],
            'input': instance['input'],
            'task_instruction': TASK_POOL[instance['seed_dataset']]['task_instruction'],
            'pos': poss,
            'neg': negs,
        })
    
    output_filename = os.path.join(args.base, 'train', args.seed_id, f'train_{args.generator_name}.jsonl')
    with open(output_filename, 'w') as file:
        for data in training_data:
            file.write(json.dumps(data) + '\n')
    
    logging.info(
        f"Done.\nOutput file is stored in {output_filename}.\n"
        f"The number of valid postive-negative samples pairs is {len(training_data)}.\n"
        f"The average number of postive samples is {sum([len(data['pos']) for data in training_data]) / len(training_data)}\n"
        f"The average number of negative samples is {sum([len(data['neg']) for data in training_data]) / len(training_data)}\n"
    )


if __name__ == '__main__':
    main()


