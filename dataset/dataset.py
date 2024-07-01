import os
import sys
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
import csv
from loguru import logger
import random
from utils.common_utils import load_pickle 
DATASET_ID_DICT = {'snli-zh':1,'sts':2,'t2-05':3,'du-10':4,'mmarco':5,'cmedqa':6}
def load_text_dataset(name, pos_dir, neg_dir, file_path, neg_K, res_data, split):
    data = []
    if split == 'train':
        hard_neg_house = load_pickle(neg_dir)
        pos_logis = load_pickle(pos_dir) 
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2'] 
            score = row['gold_label']
            if score == 'entailment':
                if split == 'train':
                    if len(hard_neg_house[text_a]) < neg_K:
                        num = math.ceil(neg_K / len(hard_neg_house[text_a]))
                        negs_logits = random.sample(hard_neg_house[text_a] * num, neg_K)
                    else:
                        negs_logits = random.sample(hard_neg_house[text_a], neg_K)
                    hardnegs, hardneg_logits = zip(*negs_logits)
                    hardnegs, hardneg_logits = list(hardnegs), list(hardneg_logits)
                elif split == 'validation':
                    hardnegs = []
                    hardneg_logits = []
                    pos_logits = []
                hardnegs = [sample[:100] for sample in hardnegs]
                data.append((text_a[:100], text_b[:100], pos_logits, hardnegs, hardneg_logits, 0))
    if split == 'train':
        split_data = data[:-10000]
        sample_num = len(split_data)
    elif split == 'validation':
        split_data = data[-10000:]  
        sample_num = len(split_data)
    res_data.extend(split_data)
    
    return res_data, sample_num


def load_sts_dataset_train(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = []
    pos_logis = load_pickle(pos_dir) 
    hard_neg_house = load_pickle(neg_dir) 
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2'] 
            if len(hard_neg_house[text_a]) < neg_K:
                num = math.ceil(neg_K / len(hard_neg_house[text_a]))
                negs_logits = random.sample(hard_neg_house[text_a] * num, neg_K)
            else:
                negs_logits = random.sample(hard_neg_house[text_a], neg_K)
            hardnegs, hardneg_logits = zip(*negs_logits)
            hardnegs, hardneg_logits = list(hardnegs), list(hardneg_logits)
            hardnegs = [sample[:100] for sample in hardnegs]
            data.append((text_a[:100], text_b[:100], pos_logits, hardnegs, hardneg_logits, 0)) 

    sample_num = len(data)
    res_data.extend(data)
    
    return res_data, sample_num

def load_sts_dataset_val(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = []
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2'] 
            data.append((text_a[:100], text_b[:100], [], [], [], 0))

    sample_num = len(data)
    res_data.extend(data)
    
    return res_data, sample_num

def load_t2_dataset_train(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = []
    pos_logis = load_pickle(pos_dir) 
    hard_neg_house = load_pickle(neg_dir) 
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2']
            if len(hard_neg_house[text_a]) < neg_K:
                num = math.ceil(neg_K / len(hard_neg_house[text_a]))
                negs_logits = random.sample(hard_neg_house[text_a] * num, neg_K)
            else:
                negs_logits = random.sample(hard_neg_house[text_a], neg_K)
            hardnegs, hardneg_logits = zip(*negs_logits)
            hardnegs, hardneg_logits = list(hardnegs), list(hardneg_logits) 
            hardnegs = [sample[:320] for sample in hardnegs]
            data.append((text_a[:50], text_b[:320], pos_logits, hardnegs, hardneg_logits, 1))

    sample_num = len(data)
    res_data.extend(data)
    
    return res_data, sample_num

def load_t2_dataset_val(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = [] 
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2']
            data.append((text_a[:50], text_b[:320], [], [], [], 1))

    sample_num = len(data)
    res_data.extend(data)
    
    return res_data, sample_num


def load_du_dataset_train(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = []
    pos_logits = load_pickle(pos_dir) 
    hard_neg_house = load_pickle(neg_dir) 
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2']
            if len(hard_neg_house[text_a]) < neg_K:
                num = math.ceil(neg_K / len(hard_neg_house[text_a]))
                negs_logits = random.sample(hard_neg_house[text_a] * num, neg_K)
            else:
                negs_logits = random.sample(hard_neg_house[text_a], neg_K)
            hardnegs, hardneg_logits = zip(*negs_logits)
            hardnegs, hardneg_logits = list(hardnegs), list(hardneg_logits) 
            hardnegs = [sample[:320] for sample in hardnegs]
            data.append((text_a[:50], text_b[:320], pos_logits, hardnegs, hardneg_logits, 1))

    sample_num = len(data)
    res_data.extend(data)
    
    return res_data, sample_num

def load_du_dataset_val(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = []
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2']
            data.append((text_a[:50], text_b[:320], [], [], [], 1))

    sample_num = len(data)
    res_data.extend(data)
    
    return res_data, sample_num

def load_mmarco_dataset_train(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = []
    pos_logis = load_pickle(pos_dir) 
    hard_neg_house = load_pickle(neg_dir) 
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2']
            if len(hard_neg_house[text_a]) < neg_K:
                num = math.ceil(neg_K / len(hard_neg_house[text_a]))
                negs_logits = random.sample(hard_neg_house[text_a] * num, neg_K)
            else:
                negs_logits = random.sample(hard_neg_house[text_a], neg_K)
            hardnegs, hardneg_logits = zip(*negs_logits)
            hardnegs, hardneg_logits = list(hardnegs), list(hardneg_logits) 
            hardnegs = [sample[:320] for sample in hardnegs]
            data.append((text_a[:50], text_b[:320], pos_logits, hardnegs, hardneg_logits, 1)) 

    sample_num = len(data)
    res_data.extend(data)
    
    return res_data, sample_num

def load_mmarco_dataset_val(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = []
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2']
            data.append((text_a[:50], text_b[:320], [], [], [], 1)) 

    sample_num = len(data)
    res_data.extend(data)
    
    return res_data, sample_num

def load_cmedqa_dataset_train(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = []
    pos_logis = load_pickle(pos_dir) 
    hard_neg_house = load_pickle(neg_dir) 
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2']
            if len(hard_neg_house[text_a]) < neg_K:
                num = math.ceil(neg_K / len(hard_neg_house[text_a]))
                negs_logits = random.sample(hard_neg_house[text_a] * num, neg_K)
            else:
                negs_logits = random.sample(hard_neg_house[text_a], neg_K)
            hardnegs, hardneg_logits = zip(*negs_logits)
            hardnegs, hardneg_logits = list(hardnegs), list(hardneg_logits) 
            hardnegs = [sample[:320] for sample in hardnegs]
            data.append((text_a[:50], text_b[:320], pos_logits, hardnegs, hardneg_logits, 1))

    sample_num = len(data)
    res_data.extend(data)
    
    return res_data, sample_num

def load_cmedqa_dataset_val(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = [] 
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2']
            data.append((text_a[:50], text_b[:320], [], [], [], 1))

    sample_num = len(data)
    res_data.extend(data)
    
    return res_data, sample_num


def collate_fn(data):
    res_s_a = []
    res_s_b = []
    res_pos_logits = []
    res_neg_K = []
    res_neg_logits = []
    res_task_id = []
    
    for d in data[0]:
        res_s_a.append(d[0])
        res_s_b.append(d[1])
        res_pos_logits.append(d[2])
        res_neg_K.append(d[3])
        res_neg_logits.extend(d[4])
        res_task_id.append(int(d[5]))

    res_neg_K = [list(group) for group in zip(*res_neg_K)]
    res_neg_K = [e for l in res_neg_K for e in l]


    return res_s_a, res_s_b, torch.FloatTensor(res_pos_logits), res_neg_K, torch.FloatTensor(res_neg_logits), torch.LongTensor(res_task_id)



class TrainDataset(Dataset):

    def __init__(self, tokenizer, pos_dir, neg_dir, datadir, names=None, batch_size=32, neg_K=8, process_index=0, num_processes=1, seed=2023):
        self.dataset_id_dict = DATASET_ID_DICT
        self.tokenizer = tokenizer
        self.data = []
        self.batch_size = batch_size
        self.sample_stas = dict()
        self.dataset_indices_range = dict()
        self.process_index = process_index
        self.num_processes = num_processes
        self.neg_K = neg_K
        self.deterministic_generator = np.random.default_rng(seed)
        names.sort(reverse=True)
        for name in names:
            if name in ['snli-zh']:
                if name == 'snli-zh':
                    start_id = len(self.data)
                    self.data, sample_num = load_text_dataset(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), os.path.join(datadir, 'PATH_TO_DATA'), self.neg_K, self.data, 'train')
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            elif name in ['sts']:
                if name == 'sts':
                    start_id = len(self.data)
                    self.data, sample_num = load_sts_dataset_train(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), datadir, self.neg_K, self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            elif name in ['t2','du', 'mmarco', 'cmedqa']:
                if name == 't2-05':
                    start_id = len(self.data)
                    self.data, sample_num = load_t2_dataset_train(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), os.path.join(datadir, 'PATH_TO_DATA'), self.neg_K, self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
                if name == 'du':
                    start_id = len(self.data)
                    self.data, sample_num = load_du_dataset_train(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), os.path.join(datadir, 'PATH_TO_DATA'), self.neg_K, self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
                if name == 'mmarco':
                    start_id = len(self.data)
                    self.data, sample_num = load_mmarco_dataset_train(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), os.path.join(datadir, 'PATH_TO_DATA'), self.neg_K, self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
                if name == 'cmedqa':
                    start_id = len(self.data)
                    self.data, sample_num = load_cmedqa_dataset_train(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), os.path.join(datadir, 'PATH_TO_DATA'), self.neg_K, self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            else:
                logger.debug('Unknown dataset: {}'.format(name))
        
        self.create_epoch()

    def __len__(self):
        return self.steps_per_epoch * self.num_processes

    def create_epoch(self):
        epoch = []
        self.steps_per_epoch = 0
        for k, v in self.dataset_indices_range.items():
            dataset_range = np.arange(*v)
            num_batches, remainer = divmod(len(dataset_range), self.batch_size * self.num_processes)
            if remainer != 0:
                dataset_range = dataset_range[:num_batches * self.batch_size * self.num_processes]
            self.deterministic_generator.shuffle(dataset_range)
            batches = dataset_range.reshape(num_batches * self.num_processes, self.batch_size).tolist()
            epoch.extend(batches)
            self.steps_per_epoch += num_batches
        self.deterministic_generator.shuffle(epoch)
        self.epoch = epoch
        self.step = 0


    def __getitem__(self, index: int):
        if self.step > (self.steps_per_epoch - 1):
            self.step = 0
        batch_indices = self.epoch[self.step*self.num_processes+self.process_index]
        batch_data = np.array(self.data)[batch_indices].tolist()
        self.step += 1

        return batch_data



class ValDataset(Dataset):

    def __init__(self, tokenizer, pos_dir, neg_dir, datadir, names=None, batch_size=32, neg_K=8, process_index=0, num_processes=1, seed=2023):
        self.dataset_id_dict = DATASET_ID_DICT
        self.tokenizer = tokenizer
        self.data = []
        self.batch_size = batch_size
        self.neg_K = neg_K
        self.sample_stas = dict()
        self.dataset_indices_range = dict()
        self.process_index = process_index
        self.num_processes = num_processes
        self.deterministic_generator = np.random.default_rng(seed)
        names.sort(reverse=True)
        for name in names:
            if name in ['snli-zh']:
                if name == 'snli-zh':
                    start_id = len(self.data)
                    self.data, sample_num = load_text_dataset(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), os.path.join(datadir, 'PATH_TO_DATA'), self.neg_K, self.data, 'validation')
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            elif name in ['sts']:
                if name == 'sts':
                    start_id = len(self.data)
                    self.data, sample_num = load_sts_dataset_val(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), os.path.join(datadir, 'PATH_TO_DATA'), self.neg_K, self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            elif name in ['t2', 'du', 'mmarco', 'cmedqa']:
                if name == 't2':
                    start_id = len(self.data)
                    self.data, sample_num = load_t2_dataset_val(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), os.path.join(datadir, 'PATH_TO_DATA'), self.neg_K, self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
                if name == 'du':
                    start_id = len(self.data)
                    self.data, sample_num = load_du_dataset_val(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), os.path.join(datadir, 'PATH_TO_DATA'), self.neg_K, self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
                if name == 'mmarco':
                    start_id = len(self.data)
                    self.data, sample_num = load_mmarco_dataset_val(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), os.path.join(datadir, 'PATH_TO_DATA'), self.neg_K, self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
                if name == 'cmedqa':
                    start_id = len(self.data)
                    self.data, sample_num = load_cmedqa_dataset_val(name, os.path.join(pos_dir, 'PATH_TO_DATA'), os.path.join(neg_dir, 'PATH_TO_DATA'), os.path.join(datadir, 'PATH_TO_DATA'), self.neg_K, self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            else:
                logger.debug('Unknown dataset: {}'.format(name))
        self.create_epoch()
        

    def __len__(self):
        return self.steps_per_epoch * self.num_processes

    def create_epoch(self):
        epoch = []
        self.steps_per_epoch = 0
        for k, v in self.dataset_indices_range.items():
            dataset_range = np.arange(*v)
            num_batches, remainer = divmod(len(dataset_range), self.batch_size * self.num_processes)
            if remainer != 0:
                dataset_range = dataset_range[:num_batches * self.batch_size * self.num_processes]
            self.deterministic_generator.shuffle(dataset_range)
            batches = dataset_range.reshape(num_batches * self.num_processes, self.batch_size).tolist()
            epoch.extend(batches)
            self.steps_per_epoch += num_batches
        self.deterministic_generator.shuffle(epoch)
        self.epoch = epoch
        self.step = 0


    def __getitem__(self, index: int):

        if self.step > self.steps_per_epoch - 1:
            self.step = 0
        batch_indices = self.epoch[self.step*self.num_processes+self.process_index]
        batch_data = np.array(self.data)[batch_indices].tolist()
        self.step += 1
        return batch_data
