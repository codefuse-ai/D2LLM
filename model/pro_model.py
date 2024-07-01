import sys
import math
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
    def forward(self, X, pad_mask):
        if self.S.dtype != torch.bfloat16:
            X = X.float()
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, pad_mask)

    def forward(self, Q, K, pad_mask=None):

        Q_ = self.fc_q(Q)
        K_, V_ = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_.split(dim_split, 2), 0) 
        K_ = torch.cat(K_.split(dim_split, 2), 0) 
        V_ = torch.cat(V_.split(dim_split, 2), 0)
        pad_mask = pad_mask.unsqueeze(1).repeat(self.num_heads, Q.size(1), 1) 
        score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V)
        score = score.masked_fill(pad_mask == 0, -1e12)
        A = torch.softmax(score, 2)
        A = A * pad_mask
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        O = Q + O
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class IEM(nn.Module): 

    def __init__(self, d_model, hidden, d_output, drop_prob=0.0):
        super(IEM, self).__init__()
        self.linear1 = nn.Linear(2*d_model, hidden)
        self.proj0 = nn.Linear(hidden, hidden)
        self.proj1 = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, d_output)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.proj0.weight)
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.sftmx = nn.Softmax(dim=-1)

    def forward(self, emb_a, emb_b):
        x = torch.cat((emb_a, emb_b), dim=-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x0 = self.proj0(x)
        x1 = self.proj1(x)
        x0 = self.relu(x0)
        x1 = self.relu(x1)
        rep = torch.stack((x0,x1),dim=0)
        logits0 = self.linear2(x0)
        logits1 = self.linear2(x1)
        logits = torch.cat((logits0, logits1), dim=-1)
        return logits, rep 



class Mymodel(nn.Module):
    def __init__(self, 
                model_name_or_path = None,
                alias = None,
                max_seq_length = 256,
                args = None
                ):
        super(Mymodel, self).__init__()
        self.alias = alias
        if self.alias == None:
            self.alias = model_name_or_path
        self.args = args
        self.max_seq_length = max_seq_length
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True, pad_token='<|endoftext|>', truncation_side='right', padding_side=self.args.padding_side)    
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.plm_model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, trust_remote_code=True) 
        self.emb_dim = self.plm_model.transformer.wte.weight.size(1)
        self.num_heads = args.num_heads
        self.ln = args.ln
        self.norm = args.norm
        self.mha_pma = PMA(self.emb_dim, self.num_heads, 1, ln=self.ln)
        self.iem = IEM(self.emb_dim, self.hidden_dim, self.output_dim)

    def forward(self, inputs_all, task_ids, mode):        
        if mode == 'train':
            output_embeddings_all = self.get_sentence_embedding(**inputs_all).reshape(2+self.args.neg_K, -1, self.emb_dim)
            output_embeddings_hardneg = output_embeddings_all[2:]
        elif mode == 'eval':
            output_embeddings_all = self.get_sentence_embedding(**inputs_all).reshape(2, -1, self.emb_dim)
        else:
            raise ValueError('Error of mode value')

        output_embeddings_a = output_embeddings_all[0]
        output_embeddings_b = output_embeddings_all[1]
        
        bs = output_embeddings_a.size(0)
        a_expand_emb = output_embeddings_a.unsqueeze(1).expand(-1, bs, -1).reshape(-1, self.emb_dim)
        b_expand_emb = output_embeddings_b.unsqueeze(0).expand(bs, -1, -1).reshape(-1, self.emb_dim)

        task_expand = task_ids.unsqueeze(1).expand(-1, bs).reshape(-1,1).squeeze()
        output_in_batch, _ = self.iem(a_expand_emb, b_expand_emb) # (bs*bs, 2)
        output_in_batch_specific_task = output_in_batch[range(task_expand.size(0)), task_expand].squeeze().reshape(bs, -1)

        if mode == 'train':
            pos_neg_emb = torch.cat([output_embeddings_b.unsqueeze(0), output_embeddings_hardneg], dim=0)
            achr_emb = output_embeddings_a.unsqueeze(0).expand(pos_neg_emb.size(0),-1,-1)
            output_hardneg, output_pos_hardneg_rep = self.iem(achr_emb, pos_neg_emb)
            task_id_gather = task_ids.unsqueeze(0).unsqueeze(-1).expand(pos_neg_emb.size(0), -1, -1)
            output_hardneg_specific_task = torch.gather(output_hardneg, -1, task_id_gather).squeeze().t()
            output_pos_hardneg_rep_specific_task = output_pos_hardneg_rep[task_ids[0]]
        elif mode == 'eval':
            output_hardneg_specific_task = None
            output_pos_hardneg_rep_specific_task = None

        return output_in_batch_specific_task, output_hardneg_specific_task, output_pos_hardneg_rep_specific_task 

    def pma_embedding(self, A, mask):
        res = self.mha_pma(A, mask).squeeze(1)
        return res

    def get_sentence_embedding(self, **inputs):
        outputs = self.plm_model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[self.keep_max_layer]
        attention_mask = inputs['attention_mask']
        res_embedding = self.pma_embedding(embedding, attention_mask)
        
        if self.norm:
            res_embedding = torch.nn.functional.normalize(res_embedding, p=2.0, dim=-1, eps=1e-12, out=None)
        return res_embedding

    def encode(self, sentences, batch_size=64, convert_to_numpy=True,
            convert_to_tensor=False, show_progress_bar=True, max_seq_length=None, **kwargs):

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        input_is_string = False        
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        with torch.no_grad():
            for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
                sentences_batch = sentences_sorted[start_index: start_index + batch_size]
                with torch.no_grad():
                    inputs = self.tokenizer(sentences_batch, padding=True, truncation=True, max_length=max_seq_length, return_tensors='pt').to(self.plm_model.device)
                    embeddings = self.get_sentence_embedding(**inputs)
                embeddings = embeddings.detach()
                if convert_to_numpy:
                    if embeddings.dtype == torch.bfloat16:
                        embeddings = embeddings.cpu().to(torch.float32)
                    else:
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

