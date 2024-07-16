from typing import Dict, List
from transformers.data.data_collator import default_data_collator
from abc import *
import torch.utils.data as data
import numpy as np
import torch
import random


class DataLoader():
    def __init__(self, examples, tokenizer,val_batch_size, max_seq_len):
        self.examples = examples
        self.tokenizer = tokenizer

        self.val_batch_size = val_batch_size
        self.max_seq_len = max_seq_len

        self.num_gpu = torch.cuda.device_count()


        self.data_collator = default_data_collator


    def _get_data_loader(self):
        dataset = PlansDataset(self.examples, self.tokenizer, self.max_seq_len)
        dataloader = data.DataLoader(dataset,
                                     batch_size=self.val_batch_size,
                                     shuffle=False,
                                     collate_fn=self.data_collator)
        return dataloader


class PlansDataset(data.Dataset):

    def __init__(self, dataset, tokenizer, max_seq_len):
        random.seed(42)

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.instances = []
        self.max_seq_len = max_seq_len

        self._form_instances()


    def _form_instances(self):
        examples = self.dataset
        batch_encoding_pos = self.tokenizer([(e[0], e[1]) for e in examples],
                                max_length=self.max_seq_len,padding="max_length", truncation=True)
        batch_encoding_neg = self.tokenizer([(e[0], e[2]) for e in examples],
                                max_length=self.max_seq_len,padding="max_length", truncation=True)
        self.instances = []
        for i in range(len(examples)):
            inputs = {'input_ids_pos': batch_encoding_pos['input_ids'][i],
                      'token_type_ids_pos': batch_encoding_pos['token_type_ids'][i],
                      'attention_mask_pos': batch_encoding_pos['attention_mask'][i],
                      'input_ids_neg': batch_encoding_neg['input_ids'][i],
                      'token_type_ids_neg': batch_encoding_neg['token_type_ids'][i],
                      'attention_mask_neg': batch_encoding_neg['attention_mask'][i]}

            self.instances.append(inputs)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]
