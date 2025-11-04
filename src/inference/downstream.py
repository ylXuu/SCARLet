''' downstream evaluation of baselines '''

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from src.synthesis.prompts import TASK_POOL
from tqdm import tqdm
from typing import List
from nltk import sent_tokenize
from collections import Counter
from rouge_score import rouge_scorer, scoring
from rouge import Rouge
import os
import re
import csv
import string
import json
import torch
import argparse
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
RETRIEVAL_PROMPT_TEMPLATE = '{task_instruction}\n\nPassages:\n{context}\n\nInput: {input}'
NORETRIEVAL_PROMPT_TEMPLATE = '{task_instruction}\n\nInput: {input}'


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# for hotpotqa
def calculate_em(output, reference):
    ''' calculate exact match recall '''
    
    def exact_presence(short_answer, context):
        ''' Verify if any of the answers is present in the given context. '''

        if normalize_answer(short_answer) in normalize_answer(context):
            return True
        
        return False
    
    assert len(output) == len(reference), f'difference size between {len(output)} and {len(reference)}'
    
    acc, hit = [], []
    for output_item, gold_item in zip(output, reference):
        acc.append(exact_presence(gold_item, output_item))
    
    # return {'str_em': 100 * np.mean(acc)}
    return 100 * np.mean(acc)


def calculate_rouge_l(output, reference):

    def rouge_l_score(prediction, ground_truth):
        rouge = Rouge()
        # no normalization
        try:
            scores = rouge.get_scores(prediction, ground_truth, avg=True)
        except ValueError:  # "Hypothesis is empty."
            return 0.0
        
        return scores['rouge-l']['f']
        
    assert len(output) == len(reference), f'difference size between {len(output)} and {len(reference)}'

    scores = []
    for output_item, gold_item in zip(output, reference):
        scores.append(rouge_l_score(output_item, gold_item))

    return 100 * np.mean(scores)


def calculate_acc(output, reference):
    acc = []
    for output_item, gold_item in zip(output, reference):
        if gold_item in output_item:
            acc.append(1.0)
        else:
            acc.append(0.0)
    
    return 100 * np.mean(acc)


def calculate_f1(output, reference):

    def _f1_score(prediction, ground_truth):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    scores = []
    for output_item, gold_item in zip(output, reference):
        scores.append(_f1_score(output_item, gold_item))
    
    return 100 * np.mean(scores)


def task_eval(dataset: str,
              outputs: List,
              references: List):
    if dataset in ['hotpotqa', 'nq', 'trex', 'zs-re']:
        return calculate_em(outputs, references)
    elif dataset in ['eli5', 'fiqa']:
        return calculate_rouge_l(outputs, references)
    elif dataset in ['fever', 'scifact', 'climate-fever']:
        return calculate_acc(outputs, references)
    elif dataset in ['wow']:
        return calculate_f1(outputs, references)


def write_result_csv(file_path: str,
                     method: str,
                     col_name: str,
                     value: float):
    cols = ['method', 'nq', 'hotpotqa', 'eli5', 'fever', 'wow', 'trex', 'zs-re', 'scifact', 'climate-fever', 'fiqa']
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=cols)
            writer.writeheader()  # Write the column names (titles)
            row_data = {key: '' for key in cols}
            row_data['method'] = method
            row_data[col_name] = value
            writer.writerow(row_data)
    else:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
        
        # Check if the specified row exists
        row_exists = False
        for row in rows:
            if row['method'] == method:  # Assume the ID field uniquely identifies a row
                row_exists = True
                # 更新该行的指定列数据
                row[col_name] = value
                break
        
        if not row_exists:
            row_data = {key: '' for key in cols}
            row_data['method'] = method
            row_data[col_name] = value
            rows.append(row_data)
        
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval', action='store_true')
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--data', required=True, type=str, help='The test data file path.')
    parser.add_argument('--generator', required=True, type=str)
    parser.add_argument('--topk', default=3, type=int)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    logging.info(f'evaluating data from {args.data}')
    with open(args.data, 'r') as f:
        # dict_keys(['_id', 'input', 'docs', 'output'])
        data = [json.loads(line) for line in f]
    
    generator = AutoModelForCausalLM.from_pretrained(args.generator, device_map='auto', torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.generator, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.retrieval:
        for instance in tqdm(data):
            docs = ['\n'.join([doc['title'], doc['text']]) for doc in instance['docs'][:args.topk]]
            context = '\n\n'.join(docs)

            messages = [
                {'role': 'system', 'content': 'You are a helpful AI assistant.'},
                {'role': 'user', 'content': RETRIEVAL_PROMPT_TEMPLATE.format(
                    task_instruction=TASK_POOL[args.dataset]['task_instruction'],
                    context=context, input=instance['input'])}
            ]
            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            chat_prompt_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)
            input_ids = torch.tensor([chat_prompt_ids], device=generator.device)
            outputs = generator.generate(input_ids, max_new_tokens=512, do_sample=False, num_return_sequences=1)[0]
            generated_texts = tokenizer.decode(outputs)
            prompt_length = len(tokenizer.decode(chat_prompt_ids))
            response = generated_texts[prompt_length:]
            instance['generated_output'] = response
    else:
        for instance in tqdm(data):
            messages = [
                {'role': 'system', 'content': 'You are a helpful AI assistant.'},
                {'role': 'user', 'content': NORETRIEVAL_PROMPT_TEMPLATE.format(
                    task_instruction=TASK_POOL[args.dataset]['task_instruction_closed_book'],
                    input=instance['input'])}
            ]
            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            chat_prompt_ids = tokenizer.encode(chat_prompt, add_special_tokens=False)
            input_ids = torch.tensor([chat_prompt_ids], device=generator.device)
            outputs = generator.generate(input_ids, max_new_tokens=512, do_sample=False, num_return_sequences=1)[0]
            generated_texts = tokenizer.decode(outputs)
            prompt_length = len(tokenizer.decode(chat_prompt_ids))
            response = generated_texts[prompt_length:]
            instance['generated_output'] = response
    
    with open(args.save_path, 'w') as file:
        for line in data:
            file.write(json.dumps(line) + '\n')
    
    results = task_eval(
        args.dataset,
        [instance['generated_output'] for instance in data],
        [instance['output'] for instance in data]
    )

    logging.info(results)
    logging.info(f'The test result of dataset {args.dataset} is {results}.')

    # write result
    write_result_csv(f'./results/{os.path.basename(os.path.normpath(args.generator))}',
                     os.path.basename(os.path.normpath(args.data)),
                     args.dataset, results)


if __name__ == '__main__':
    main()
