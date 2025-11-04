
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
)
from torch.utils.data import Dataset
import json
import argparse
import torch
import torch.nn.functional as F


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


class RetrieverDataCollator:
    def __init__(self,
                 tokenizer,
                 padding=True,
                 truncation=True,
                 max_length=512):
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
    
    def __call__(self, features):
        """
        The core logic of the Collator is to receive a batch of samples (features) and return a populated batch.

        :param features: 
            A list of samples passed from `DataLoader`, where each sample is a dictionary.
        
        :return: The populated batch, usually a dictionary.
        """

        # Extract the `input_ids` and `attention_mask` from each sample.
        # print(len(features))
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]

        # Use a tokenizer for batch processing and piling/truncating.
        batch = self.tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return batch


class RetrieverDataset(Dataset):
    def __init__(self, data, tokenizer, strategy, max_length=512):
        self.data = data # dict_keys(['_id', 'task_name', 'seed_dataset', 'input', 'task_instruction', 'pos', 'neg'])
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        query = item['input']
        inst = item['task_instruction']
        
        if self.strategy == 'lists':
            poss = [passage['text'] for passage in item['pos']]
            negs = [passage['text'] for passage in item['neg']]
        elif self.strategy == 'top_1':
            poss = [item['pos'][0]['text']]
            negs = [passage['text'] for passage in item['neg']]
        
        inputs = []
        labels = []

        # top_1策略
        inputs.append(self.tokenizer(inst + '[SEP]' + query,
                                     poss[0],
                                     truncation=True,
                                     padding='max_length',
                                     max_length=self.max_length))
        labels.append(0)
        
        for neg in negs:
            inputs.append(self.tokenizer(inst + '[SEP]' + query,
                                         neg,
                                         truncation=True,
                                         padding='max_length',
                                         max_length=self.max_length))
            labels.append(0)

        return {
            'input_ids': torch.stack([torch.tensor(inp['input_ids']) for inp in inputs]),
            'attention_mask': torch.stack([torch.tensor(inp['attention_mask']) for inp in inputs]),
            # 'labels': torch.tensor(labels)
        }
        


class RetrieverTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids'] # (batch_size, num_docs, seq_len)
        attention_mask = inputs['attention_mask']
        # labels = inputs['labels'] # (batch_size, N_pos + N_neg)

        batch_size, num_docs, _ = input_ids.shape

        # Flatten (batch_size, num_docs, seq_len) to (batch_size * num_docs, seq_len).
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        
        # Get the model's logits, and reshape to (batch_size, num_docs).
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.view(batch_size, num_docs)
        
        # Set the logits of the first document (positive example) as the target.
        target = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # Use CrossEntropyLoss to calculate the loss.
        loss = F.cross_entropy(logits, target)
        
        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str, help='The training data file path.')
    parser.add_argument('--model_path', required=True, type=str, help='The model path.')
    parser.add_argument('--strategy', required=True, type=str, help='The sampling strategy.')
    parser.add_argument('--ckpt_path', default='./ckpt', type=str, help='The path to store checkpoints.')
    parser.add_argument('--save_path', default='./ckpt/models', type=str, help='The path of saving final model.')
    parser.add_argument('--log_path', default='./logs', type=str, help='The path to store logs.')
    args = parser.parse_args()

    data = load_data(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = RetrieverDataset(data, tokenizer, args.strategy)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=1)
    # collator = RetrieverDataCollator(tokenizer, padding=True, truncation=True, max_length=512)
    
    training_args = TrainingArguments(
        output_dir=args.ckpt_path,
        evaluation_strategy='no',
        learning_rate=6e-5,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir=args.log_path,
        logging_steps=10,
        save_steps=1000,
        # max_grad_norm=1.0,
    )
    trainer = RetrieverTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        # data_collator=collator
    )

    trainer.train()

    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)


if __name__ == '__main__':
    main()

