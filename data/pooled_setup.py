''' combine all corpus under data directory, and the pooled corpus '''


import json

corpus_list = [
    'wikipedia_100_2019_08_01.jsonl',
    'scifact/corpus.jsonl',
    'climate-fever/corpus.jsonl',
    'fiqa/corpus.jsonl',
]

def main():
    data = []
    for corpus_file in corpus_list:
        with open(corpus_file, 'r', encoding='utf8') as file:
            for line in file:
                item = json.loads(line)
                data.append(item)
    
    output_file = 'pooled_corpus.jsonl'
    with open(output_file, 'w') as file:
        for line in data:
            file.write(json.dumps(line) + '\n')

if __name__ == '__main__':
    main()

