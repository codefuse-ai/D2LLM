import os
import sys
import torch
import argparse
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from utils.common_utils import load_pickle, write_pickle


def sts_template_v5(text1, text2):
    return f'#P和#H将分别描述一种事件或问题，它们可能并无关系。仅使用此描述和您对世界的了解，判断#H是不是一个关于#P中的事件绝对正确的句子，或者#H是不是绝对正确地描述了#P的事件或问题，请回答是或不是，若您不确定，请回答不是。\n#P：{text1}\n#H：{text2}\n回答：'

def context_template_v5(text1, text2):
    return f'#Q将描述一个问题，#A将描述一个网络段落，它们可能并没有关系。仅依据这些描述和您对世界的了解，判断#A能不能正确地回答#Q中提出的问题，请回答能或不能。\n#Q：{text1}\n#A：{text2}\n回答：'


def generate_logits(model_dir, neg_pkl_file, task_type, bs, teacher_max_seq_length, num_shards, id_shard):
    bm_25_dict = load_pickle(neg_pkl_file)
    all_sample_list = []
    len_dict = {}
    all_logits = []
    res_dict = {}
    lenth_one = len(list(bm_25_dict.keys()))/num_shards
    for i, query in enumerate(bm_25_dict):
        if i >= lenth_one*id_shard and i < lenth_one*(id_shard+1):
            doc_list = bm_25_dict[query]
            len_dict[i] = len(doc_list)
            if task_type == 'context':
                    qry_doc_list = [context_template_v5(query, d) for d in doc_list]
            elif task_type == 'sts':
                    qry_doc_list = [sts_template_v5(query, d) for d in doc_list]
            all_sample_list.extend(qry_doc_list)
    teacher_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, pad_token='<|endoftext|>', truncation_side='right', padding_side='left')
    teacher_tokenizer.pad_token_id = teacher_tokenizer.eod_id
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to('cuda') 
    model.eval()
    if task_type == 'sts':
        yes_id = teacher_tokenizer.encode('是')[0]
        no_id = teacher_tokenizer.encode('不是')[0]
    elif task_type == 'context':
        yes_id = teacher_tokenizer.encode('能')[0]
        no_id = teacher_tokenizer.encode('不能')[0]
    else:
        raise ValueError(f'Error: No Task Type {task_type}')
    with torch.no_grad():
        for start_index in trange(0, len(all_sample_list), bs, disable=False):
            print(start_index)
            cross_sentence_batch = all_sample_list[start_index: start_index+bs]
            cross_sentence_inputs = teacher_tokenizer(text=cross_sentence_batch, padding='max_length', max_length=teacher_max_seq_length, truncation=True, return_tensors='pt').to('cuda')
            outputs_logits = model(**cross_sentence_inputs).logits
            outputs_logits = outputs_logits[:, -1, [yes_id, no_id]].cpu().float().numpy().tolist()
            all_logits.extend(outputs_logits)
    assert len(all_logits) == len(all_sample_list)
    start = 0
    for i, query in enumerate(bm_25_dict):
        if i >= lenth_one*id_shard and i < lenth_one*(id_shard+1):
            end = start + len_dict[i]
            doc_list = bm_25_dict[query]
            logits_list = all_logits[start:end]
            assert len(doc_list) == len(logits_list)
            res_doc_logits = list(zip(doc_list, logits_list))
            res_dict[query] = res_doc_logits
            start = end
    return res_dict 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='', type=str)
    parser.add_argument('--hardneg_dir', default='', type=str)
    parser.add_argument('--output_pkl', default='', type=str)
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--task_type', default='', type=str)
    parser.add_argument('--bs', default=140, type=int)
    parser.add_argument('--K', type=int)
    parser.add_argument('--teacher_max_seq_length', default=500, type=int)
    parser.add_argument('--num_shards', default=8, type=int)
    parser.add_argument('--id_shard', default=0, type=int)
    args = parser.parse_args()

    neg_pkl_file = args.hardneg_dir
    output_pkl_path = args.output_pkl
    res_dict = generate_logits(args.model_dir, neg_pkl_file, args.task_type, args.bs, args.teacher_max_seq_length, args.num_shards, args.id_shard)
    write_pickle(res_dict, output_pkl_path)

