import sys
import csv
import time
import os
import jieba
import pickle
import argparse
from rank_bm25 import BM25Okapi
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset

def write_pickle(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj


def load_snli_zh(path):
    queries = []
    corpus = []

    pos_sample_dict = defaultdict(list)

    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentence1'] 
            text_b = row['sentence2'] 
            label = row['gold_label']
            
            if isinstance(text_b, str):
                corpus.append(text_b)

            if label == 'entailment':
                if isinstance(text_a, str):
                    queries.append(text_a)

                pos_sample_dict[text_a].append(text_b)

    return queries, list(set(corpus)), pos_sample_dict


def load_sts_zh(path):
    queries = []
    corpus = []
    pos_sample_dict = defaultdict(list)
    dataset = load_dataset(path, split='train')
    for id, row in enumerate(dataset):
        text_a = row['sentence1'] 
        text_b = row['sentence2'] 
        label = row['label']
        if isinstance(text_b, str):
            corpus.append(text_b)
        if path.split('/')[-1] != 'STS-B':
            if label == 1:
                if isinstance(text_a, str):
                    queries.append(text_a)

                pos_sample_dict[text_a].append(text_b)
        else:
            if label >= 4:
                if isinstance(text_a, str) :
                    queries.append(text_a)
                pos_sample_dict[text_a].append(text_b)
    return queries, list(set(corpus)), pos_sample_dict

def load_t2(path):
    queries = []
    corpus = []

    pos_sample_dict = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            text_a = row['sentece1'] 
            text_b = row['sentence2'] 
            
            if isinstance(text_b, str):
                corpus.append(text_b[:320])

            if isinstance(text_a, str):
                queries.append(text_a)

            pos_sample_dict[text_a].append(text_b[:320])


            
    return queries, list(set(corpus)), pos_sample_dict



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='', type=str)
    parser.add_argument('--K', default=10, type=int)
    parser.add_argument('--num', default=50, type=int)
    
    args = parser.parse_args()


    stopwords = []
    with open('STOPWORDS_PATH', 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip('\n')
            stopwords.append(line)
    output_dir = "OUTPUTS_NEG_BM25_PATH"
    if args.data_name == 'snli-zh':
        queries, corpus, pos_sample_dict = load_snli_zh("NLI_DATA_PATH")
        output_pickle = os.path.join(output_dir, args.data_name+'.pkl')
    if args.data_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']:
        queries, corpus, pos_sample_dict = load_sts_zh("STS_DATA_PATH")
        output_pickle = os.path.join(output_dir, args.data_name+'.pkl')
    if args.data_name == 't2':
        queries, corpus, pos_sample_dict = load_t2("T2_DATA_PATH")
        output_pickle = os.path.join(output_dir, args.data_name+'.pkl')
    tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
    tokenized_corpus = [list(set(tokenized_doc).difference(set(stopwords))) for tokenized_doc in tokenized_corpus]
    bm25 = BM25Okapi(tokenized_corpus)


    tokenized_queries = [list(jieba.cut(q)) for q in queries]
    tokenized_queries = [list(set(tokenized_query).difference(set(stopwords))) for tokenized_query in tokenized_queries]
    assert len(queries) == len(tokenized_queries)

    hard_neg_sample_dict = defaultdict(list)
    for i,tokenized_query in enumerate(tqdm(tokenized_queries)):
        doc_scores = bm25.get_scores(tokenized_query)
        res_docs = bm25.get_top_n(tokenized_query, corpus, n=args.K)
        for pos in pos_sample_dict[queries[i]]:
            while pos in res_docs:
                res_docs.remove(pos)

        hard_neg_sample_dict[queries[i]] = res_docs
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    write_pickle(hard_neg_sample_dict, output_pickle)
    