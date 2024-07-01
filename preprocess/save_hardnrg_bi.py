import sys
import os
import csv
import pathlib
import json
import argparse
import warnings
import deepspeed
from enum import Enum
from typing import Union, List
from datasets import load_dataset
from tqdm import tqdm, trange
from collections import defaultdict
from utils.common_utils import *
warnings.filterwarnings('ignore')
from mteb.mteb import MTEB
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def makedirs(path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path

class EncoderType(Enum):
    FIRST_LAST_AVG = 0
    LAST_AVG = 1
    CLS = 2
    POOLER = 3
    MEAN = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError()

class BaseBertModel:
    def __init__(
        self, 
        model_name_or_path = None,
        max_seq_length = 512,
        encoder_type = 'CLS',
        alias = None
    ):
        self.model_name_or_path = model_name_or_path
        encoder_type = EncoderType.from_string(encoder_type) if isinstance(encoder_type, str) else encoder_type
        if encoder_type not in list(EncoderType):
            raise ValueError(f'encoder_type must be in {list(EncoderType)}')
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, truncation_side='right', padding_side='right')
        self.plm_model = AutoModel.from_pretrained(model_name_or_path)
        self.results = {} 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.plm_model.to(self.device)

    

    def get_sentence_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        model_output = self.plm_model(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.encoder_type == EncoderType.FIRST_LAST_AVG:
            first = model_output.hidden_states[1]
            last = model_output.hidden_states[-1]
            seq_length = first.size(1) 

            first_avg = torch.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1) 
            last_avg = torch.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)  
            final_encoding = torch.avg_pool1d(
                torch.cat([first_avg.unsqueeze(1), last_avg.unsqueeze(1)], dim=1).transpose(1, 2),
                kernel_size=2).squeeze(-1)
            return final_encoding

        if self.encoder_type == EncoderType.LAST_AVG:
            sequence_output = model_output.last_hidden_state  
            seq_length = sequence_output.size(1)
            final_encoding = torch.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding

        if self.encoder_type == EncoderType.CLS:
            sequence_output = model_output.last_hidden_state
            return sequence_output[:, 0] 

        if self.encoder_type == EncoderType.POOLER:
            return model_output.pooler_output 

        if self.encoder_type == EncoderType.MEAN:
            token_embeddings = model_output.last_hidden_state  # Contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            final_encoding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9)
            return final_encoding  # [batch, hid_size]

    def batch_to_device(self, batch, device):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        return batch


    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = True,
            max_seq_length: int = None,
    ):
        self.plm_model.eval()
        if device is None:
            device = self.device
        self.plm_model.to(device)

        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        if convert_to_tensor:
            convert_to_numpy = False
        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            with torch.no_grad():
                features = self.tokenizer(
                    sentences_batch, max_length=max_seq_length,
                    padding=True, truncation=True, return_tensors='pt'
                )
                features = self.batch_to_device(features, device)
                embeddings = self.get_sentence_embeddings(**features)
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                if convert_to_numpy:
                    embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

def write_t2_corpus(model, output_dir):
    makedirs(output_dir)
    corpus = set()
    corpus_path = "PATH_TO_SAVED_CORPUS"
    with open(corpus_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            corpus.add(row['text'][:320])
    
    corpus = list(corpus)
    
    corpus_psg_id_dict = {psg:id for id, psg in enumerate(corpus)} 
    corpus_id_psg_dict = {id:psg for id, psg in enumerate(corpus)} 

    corpus_psg_id_dict_path = "PATH_TO_SAVED_PSG_ID_DICT"
    corpus_id_psg_dict_path = "PATH_TO_SAVED_ID_PSG_DICT"
    corpus_rep_path = "PATH_TO_SAVED_REP"
    corpus_rep = model.encode(corpus, batch_size=1500, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True, max_seq_length=250).to('cpu')
    
    write_pickle(corpus_psg_id_dict, corpus_psg_id_dict_path)  
    write_pickle(corpus_id_psg_dict, corpus_id_psg_dict_path)  
    write_pickle(corpus_rep, corpus_rep_path)  

      
def write_t2_qry(model, corpus_psg_id_dict_path, corpus_id_psg_dict_path, corpus_rep_path, output_dir, K):
    res = defaultdict(list)
    queries = []
    pos_sample_dict = defaultdict(list)
    corpus_psg_id_dict = load_pickle(corpus_psg_id_dict_path)
    corpus_id_psg_dict = load_pickle(corpus_id_psg_dict_path)
    corpus_rep = load_pickle(corpus_rep_path)
    query_path = f'DATA_PATH'
    data_all_path = f'ALL_DATA_PATH'

    with open(data_all_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2'][:320] 
            pos_sample_dict[text_a].append(text_b)

    with open(query_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            queries.append(text_a)
    
    makedirs("QUERY_PATH")
    if not os.path.exists("QUERY_PKL_PATH"):
        queries_rep = model.encode(queries, batch_size=1500, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True, max_seq_length=100).to('cpu')
        write_pickle(queries_rep, "QUERY_PKL_PATH")
    queries_rep = load_pickle("QUERY_PKL_PATH")


    qry_chunk_size = 2000
    qry_num = queries_rep.size(0)
    corpus_num = corpus_rep.size(0)
    for start in trange(0, qry_num, qry_chunk_size, disable=False):
        end = min(start+qry_chunk_size, qry_num)
        qry_bch_rep = queries_rep[start:end, :]
        score_bch = cos_sim(qry_bch_rep, corpus_rep)
        _, ids = torch.topk(score_bch, min(K+1, score_bch.size(1)), dim=1, largest=True,sorted=True)
        ids = ids.tolist()
        for qry_id in range(start, end):
            id_from_zero = qry_id - start
            qry_text = queries[qry_id]
            pos_text_list = pos_sample_dict[qry_text]
            for sub_id in ids[id_from_zero][-100:]:
                hardneg_text = corpus_id_psg_dict[sub_id]
                if hardneg_text not in pos_text_list and hardneg_text not in res[qry_text]:
                     res[qry_text].append(hardneg_text)
    
    res_path = "FINAL_RES_PATH"
    write_pickle(res, res_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--data_path', default='', type=str) 
    parser.add_argument('--output_dir', default='', type=str) 
    parser.add_argument('--ratio', default=0.5, type=float) 
    parser.add_argument('--K', default=100, type=int) 
    parser.add_argument('--base_model_dir', default='', type=str)
    parser.add_argument('--max_seq_len', default=250, type=int, help='max sequence length') 
    parser.add_argument('--seed', default=2023, type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_corpus_path = os.path.join(args.data_path, 'corpus')
    makedirs(args.output_corpus_path)
    model = BaseBertModel(model_name_or_path=args.base_model_dir,
        alias=None,
        encoder_type = 'CLS',
        max_seq_length=args.max_seq_len)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.plm_model.to(device)
    model.plm_model.eval()

    if not os.path.exists(os.path.join(f'{args.output_corpus_path}', 'corpus_rep.pkl')):
        if args.dataset == 'T2Ranking':
            write_t2_corpus(model, f'{args.output_corpus_path}')
    corpus_psg_id_dict_path = os.path.join(f'{args.output_corpus_path}', 'corpus_psg_id_dict.pkl')
    corpus_id_psg_dict_path = os.path.join(f'{args.output_corpus_path}', 'corpus_id_psg_dict.pkl')
    corpus_rep_path = os.path.join(f'{args.output_corpus_path}', 'corpus_rep.pkl')
    if args.dataset == 'T2Ranking':
        write_t2_qry(args.ratio, model, corpus_psg_id_dict_path, corpus_id_psg_dict_path, corpus_rep_path, args.output_dir, args.K)


if __name__ == '__main__':
    main()