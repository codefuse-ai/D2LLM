import sys
import os
import warnings
import json
import logging
import argparse
import random
import time
import tracemalloc
from collections import defaultdict
from copy import deepcopy
import deepspeed
import transformers
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import *
from model.pro_model import *
from utils.common_utils import *
logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings('ignore')


def cal_loss_in_batch(args, student_logits, temperature, criterion):

    bs = student_logits.size(0)
    logits = student_logits/temperature
    labels = torch.arange(bs, device=logits.device)
    loss_bs = criterion(logits, labels) 
    
    return (loss_bs.sum())/ (bs * bs)


def cal_loss_hardneg(args, teacher_logits, student_logits, temperature_teacher, temperature, nll_criterion):

    loss_hardneg_weight = args.alpha

    def softmax(X, temp):
        X = (X/temp).exp()
        res = X / (X.sum(-1, keepdims=True)+1e-20)
        return res

    bs = teacher_logits.size(0)
    neg_K = teacher_logits.size(1)-1
    teacher_logits = softmax(teacher_logits, temperature_teacher)[:,:, 0]
    teacher_logits[:, 1:] = 1 - teacher_logits[:, 1:]
    inputs = (softmax(student_logits*teacher_logits, temperature)).log()
    labels = torch.zeros(bs, dtype=torch.long, device=student_logits.device)
    loss_bs = nll_criterion(inputs, labels)


    loss_bs = loss_bs * loss_hardneg_weight
    return loss_bs.sum() / (bs * neg_K)


def cal_loss_rd(args, teacher_logits, student_logits, teacher_temperature):

    loss_pearson_weight = args.beta

    def softmax(X, temp):
        X = (X/temp).exp()
        res = X / (X.sum(-1, keepdims=True)+1e-20)
        return res

    def pearsonr(x,y,batch_first=True):
        assert x.shape == y.shape
        if batch_first:
            dim = -1
        else:
            dim = 0
        assert x.shape[dim] > 1
        centered_x = x - x.mean(dim=dim, keepdim=True)
        centered_y = y - y.mean(dim=dim, keepdim=True)
        covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
        bessel_corrected_covariance = covariance / (x.shape[dim] - 1)
        x_std = x.std(dim=dim, keepdim=True)
        y_std = y.std(dim=dim, keepdim=True)
        corr = bessel_corrected_covariance / ((x_std * y_std)+1e-8)
        return corr
    


    bs = student_logits.size(0)
    teacher_logits = softmax(teacher_logits, teacher_temperature)[:,:, 0] 
    spearson = pearsonr(student_logits, teacher_logits).squeeze()

    loss_bs = 1 - spearson

    loss_bs = loss_bs * loss_pearson_weight

    return loss_bs.sum() / bs
    


def cal_loss_rd2(args, teacher_logits_pos_hardneg, teacher_logits_pos_inbatch, teacher_temperature, student_logits_pos_hardneg, student_logits_pos_inbatch, sigmoid, scale_param):

    loss_bpr_weight = args.gamma

    def softmax(X, temp):
        X = (X/temp).exp()
        res = X / (X.sum(-1, keepdims=True)+1e-20)
        return res


    teacher_logits_pos_hardneg = softmax(teacher_logits_pos_hardneg, teacher_temperature)[:,:, 0] 
    teacher_logits_pos_inbatch = softmax(teacher_logits_pos_inbatch, teacher_temperature)[:,:, 0] 

    bs = student_logits_pos_hardneg.size(0) 
    neg_K = student_logits_pos_hardneg.size(1)-1 
    inbatch = student_logits_pos_inbatch.size(1)-1 
    student_logits_hardneg = student_logits_pos_hardneg[:, 1:] 
    eye = torch.eye(bs, dtype=torch.bool)
    student_logits_inbatch = student_logits_pos_inbatch[~eye].reshape(bs, -1) 
    loss_hardneg_inbatch = -((sigmoid(student_logits_hardneg.view(bs, neg_K, 1).expand(-1, -1, inbatch).reshape(bs, -1) - student_logits_inbatch.unsqueeze(1).expand(-1, neg_K,-1).reshape(bs, -1))+1e-8).log()) 
    weight_hardneg_inbatch = teacher_logits_hardneg.repeat_interleave(inbatch, dim=1) - teacher_logits_inbatch.repeat((1, neg_K))  
    weight_hardneg_inbatch = torch.clamp(weight_hardneg_inbatch, min=0) / scale_param
    loss_bs = (loss_hardneg_inbatch * weight_hardneg_inbatch).sum(-1)
    loss_bs = loss_bs * loss_bpr_weight

    return loss_bs.sum() / (bs * neg_K * inbatch)


def cal_feat_loss(args, teacher_feat_cos, student_feature_pos_hardneg):

    loss_feat_weight = args.eta
    neg_K = teacher_feat_cos.size(1)
    student_feature_pos_hardneg = student_feature_pos_hardneg.transpose(0, 1)
    student_feature_pos_hardneg = student_feature_pos_hardneg / student_feature_pos_hardneg.norm(dim=-1, keepdim=True)
    student_feat_cos = torch.matmul(student_feature_pos_hardneg, student_feature_pos_hardneg.transpose(-2, -1))
    loss_bs = ((teacher_feat_cos - student_feat_cos) ** 2).sum((-1,-2))

    loss_bs = loss_bs * loss_feat_weight

    return loss_bs.sum() / (neg_K * neg_K)


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_dir', default='/mnt/user/415350/download_models/Qwen-7B-Chat', type=str, help='Model directory')
    parser.add_argument('--train_data_list', nargs='+')
    parser.add_argument('--pos_dir', default='PATH_TO_POS_LOGITS', type=str)
    parser.add_argument('--neg_dir', default='PATH_TO_HARDNEG_LOGITS', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--inbatch_pkl_path_dir', default='PATH_TO_INBATCH_LOGITS_PKL')
    parser.add_argument('--feature_pkl_path_dir', default='PATH_TO_FEATURE_PKL')
    parser.add_argument('--batch_size', default=32, type=int, help='bs')
    parser.add_argument('--neg_K', default=8, type=int, help='num of hard negs')
    parser.add_argument('--num_heads', default=32, type=int, help='num_heads of pma')
    parser.add_argument('--hidden_dim', default=512, type=int, help='hidden dim of my mlp')
    parser.add_argument('--output_dim', default=1, type=int, help='output dim of my mlp')
    parser.add_argument('--ln', default=True, type=str2bool, help='layer norm for pma')
    parser.add_argument('--norm', default=False, type=str2bool, help='norm after sentence pooling')
    parser.add_argument('--num_epochs', default=5, type=int, help='training epochs')
    parser.add_argument('--padding_side', default='right', type=str, help='padding side')
    parser.add_argument('--max_seq_length', default=250, type=int, help='max_seq_len') 
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--alpha', default=1, type=float, help='trade-off param') 
    parser.add_argument('--beta', default=1, type=float, help='trade-off param') 
    parser.add_argument('--gamma', default=0.01, type=float, help='trade-off param') 
    parser.add_argument('--eta', default=0.001, type=float, help='trade-off param') 
    parser.add_argument('--temperature_in_batch', default=1, type=float, help='temperature in in-batch')
    parser.add_argument('--temperature_hardneg', default=1, type=float, help='temperature in hardneg')
    parser.add_argument('--temperature_teacher_hardneg', default=1, type=float, help='temperature in teacher logits')
    parser.add_argument('--scale_param', default=1, type=float, help='scale param')
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--eval_interval', default=200, type=int)
    parser.add_argument('--tb_dir', default='PATH_TO_TENSORBOARD_PATH', type=str)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--num_ckpt', default=5, type=int)
    parser.add_argument('--training_log', default='PATH_TO_TRAINING_LOG')
    parser.add_argument('--output_dir', default='PATH_TO_OUTPUT_MODEL', type=str, help='Model output directory')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    parser.add_argument('--gradient_clipping', default=1.0, type=float, help='max_grad_norm')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--bf16', default=True, type=str2bool)
    parser.add_argument('--verbose', default=True, type=str2bool)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--local_rank', type=int, default=-1, help='ds')
    parser.add_argument('--global_rank', type=int, default=-1, help='ds')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args.world_size = int(os.getenv('WORLD_SIZE', '0'))
    
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=f'{arg.training_log}'))
    
    if args.seed is not None:
        set_seed(args.seed)
        transformers.set_seed(args.seed)

    micro_bs = args.batch_size

    model = Mymodel(model_name_or_path=args.base_model_dir,
    alias=None,
    max_seq_length=args.max_seq_length,
    args=args)
    model.plm_model.gradient_checkpointing_enable()

    summary_writer = SummaryWriter(log_dir=args.tb_dir)

    train_data_flag = False
    lora_config = LoraConfig(
                            r=8,
                            lora_alpha=8,
                            target_modules=['c_attn', 'c_proj', 'w1', 'w2'],
                            layers_to_transform=list(range(0, 32)),
                            lora_dropout=0.05,
                            bias="none",
                            inference_mode=False,
                            task_type=TaskType.CAUSAL_LM
                            )
    model.plm_model = get_peft_model(model.plm_model, lora_config)
    
    update_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_optimizer = list([(n,p) for n,p in model.named_parameters() if p.requires_grad])
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'lr': args.lr, 'weight_decay': args.weight_decay, 'betas': [0.8,0.999], 'eps': 1e-6, 'name':'d'},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'lr': args.lr, 'weight_decay': 0.0, 'betas': [0.8,0.999], 'eps': 1e-6, 'name':'nd'}]

    ds_config = {
    "bfloat16": {
        "enabled": args.bf16
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "gradient_clipping": args.gradient_clipping,
    "train_batch_size": args.world_size,
    "train_micro_batch_size_per_gpu": 1,
    "steps_per_print": 1e5
}
    
    fake_bs = ds_config['train_micro_batch_size_per_gpu']
    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(optimizer_grouped_parameters)
    scheduler = deepspeed.runtime.lr_schedules.WarmupLR(optimizer, warmup_min_lr=[0,0], warmup_max_lr=[args.lr,args.lr],
                     warmup_num_steps=1000)

    model_engine, optimizer, _, scheduler = deepspeed.initialize(args=args, model=model, model_parameters=update_parameters, optimizer=optimizer, lr_scheduler=scheduler, config=ds_config)
    device = torch.device(args.local_rank)
    args.device = device
    args.global_rank = torch.distributed.get_rank()

    train_dataset = TrainDataset(model.tokenizer, pos_dir=args.pos_dir, neg_dir=args.neg_dir, datadir=args.data_dir, names=args.train_data_list, batch_size=micro_bs, neg_K=args.neg_K, process_index=args.global_rank, num_processes=args.world_size)
    val_dataset = ValDataset(model.tokenizer, pos_dir=args.pos_dir, neg_dir=args.neg_dir, datadir=args.data_dir, names=args.train_data_list, batch_size=micro_bs, neg_K=args.neg_K, process_index=args.global_rank, num_processes=args.world_size)

    if args.global_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = RandomSampler(val_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=fake_bs, shuffle=False, sampler=train_sampler,collate_fn=collate_fn, num_workers=0) 
    val_dataloader = DataLoader(val_dataset, batch_size=fake_bs, shuffle=False, sampler=val_sampler,collate_fn=collate_fn, num_workers=0) 
    if len(train_dataset) > 0:
        train_data_flag = True
    
    if not train_data_flag:
        raise ValueError("Error, train_file|use_hf_dataset must be specified")
    
    all_dataset_id = train_dataset.dataset_id_dict
    all_dataset_id_reverse = {v:k for k, v in train_dataset.dataset_id_dict.items()}
    rel_dataset_id = [all_dataset_id[dataset_name] for dataset_name in args.train_data_list]
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader_size = len(train_dataloader)
    val_loader_size = len(val_dataloader)
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    nll_criterion = nn.NLLLoss(reduction='none')

    global_step = 0
    best_eval_metric = 0
    trained_epochs = 0
    min_reduce_loss_eval = float('inf')
    best_epoch = 0
    stop = 0

    teacher_feature_cos_dict = load_pickle(args.feature_pkl_path_dir)
    teacher_inbatch = load_pickle(args.inbatch_pkl_path_dir)

    reduce_loss = 0
    reduce_loss_eval = 0
    reduce_loss_in_batch = 0
    reduce_loss_in_batch_eval = 0
    reduce_loss_hardneg = 0
    reduce_loss_rd = 0
    reduce_loss_rd2 = 0
    reduce_loss_feat = 0
    reduce_inbatch_sample_num = {}


    for current_epoch in trange(int(args.num_epochs), desc="Epoch", disable=(args.global_rank!=0), mininterval=0):
        if stop >= args.patience:
            logging.info(f'Early Stop at {current_epoch+1}-th epoch {global_step}-th step')
            logging.info(f'Model trained!\nThe best model at {best_epoch+1}-th epoch {best_step}-th step')
            break
        torch.cuda.empty_cache()
        model_engine.train()

        loss_epoch_eval = 0 
        
        batch_iterator = tqdm(train_dataloader,
                                desc=f"Running Epoch {current_epoch + 1} of {args.num_epochs}",
                                disable=(args.global_rank!=0),
                                mininterval=0)
        for step, batch in enumerate(batch_iterator):
            sentence_a, sentence_b, logits_teacher_pos, sentence_hardneg, logits_teacher_hardneg, task_id = batch
            sentence_all = sentence_a + sentence_b + sentence_hardneg
            bs = logits_teacher_pos.size(0)
            key = 'global_rank' + str(args.global_rank)
            logits_teacher_inbatch = teacher_logits_dict[key][step].to(device)
            feature_teacher_cos = teacher_feature_cos_dict[key][step].to(device)
            
            inputs_all = model.tokenizer(sentence_all, padding='max_length', max_length=args.max_seq_length, truncation=True, return_tensors='pt')
            inputs_all = inputs_all.to(device)
            task_id = task_id.to(device)
            logits_student_in_batch, logits_student_hardneg, rep_student_pos_hardneg = model_engine(inputs_all, task_id, 'train')

            loss_in_batch = cal_loss_in_batch(args, logits_student_in_batch, args.temperature_in_batch, criterion)
            logits_teacher_pos = logits_teacher_pos.to(args.device)
            logits_teacher_hardneg = logits_teacher_hardneg.reshape(micro_bs, args.neg_K, 2).to(args.device)
            logits_teacher_hardneg = torch.cat([logits_teacher_pos.unsqueeze(1), logits_teacher_hardneg], dim=1)
            loss_hardneg = cal_loss_hardneg(args, logits_teacher_hardneg, logits_student_hardneg, args.temperature_teacher_hardneg, args.temperature_hardneg, nll_criterion) 
            
            loss_rd = cal_loss_rd(args, logits_teacher_hardneg, logits_student_hardneg, args.temperature_teacher_hardneg)

            loss_rd2 = cal_loss_rd2(args, logits_teacher_hardneg, logits_teacher_inbatch, args.temperature_teacher_hardneg, logits_student_hardneg, logits_student_in_batch, sigmoid, args.scale_param)

            loss_feat = cal_feat_loss(args, feature_teacher_cos, rep_student_pos_hardneg)

            loss_batch = loss_in_batch + loss_hardneg + loss_outer_rd + loss_rd + loss_feat 
            if args.verbose:
                batch_iterator.set_description(
                    f"Epoch: {current_epoch + 1}/{args.num_epochs}, Batch:{step}/{len(train_dataloader)}, Loss: {loss_batch:9.4f}")

            model_engine.backward(loss_batch)
            model_engine.step()
            
            if (step + 1) % args.gradient_accumulation_steps == 0: 
                global_step += 1

            reduce_loss += loss_batch.detach()
            reduce_loss_in_batch += loss_in_batch.detach()
            reduce_loss_hardneg += loss_hardneg.detach()
            reduce_loss_rd += loss_rd.detach()
            reduce_loss_rd2 += loss_rd2.detach()
            reduce_loss_feat += loss_feat.detach()
                
            if global_step % args.log_interval == 0:
                dist.all_reduce(reduce_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(reduce_loss_in_batch, op=dist.ReduceOp.SUM)
                dist.all_reduce(reduce_loss_hardneg, op=dist.ReduceOp.SUM)
                dist.all_reduce(reduce_loss_rd, op=dist.ReduceOp.SUM)
                dist.all_reduce(reduce_loss_rd2, op=dist.ReduceOp.SUM)
                dist.all_reduce(reduce_loss_feat, op=dist.ReduceOp.SUM)

                reduce_loss = reduce_loss.item() / (args.gradient_accumulation_steps * args.log_interval * args.world_size)
                reduce_loss_in_batch = reduce_loss_in_batch.item() / (args.gradient_accumulation_steps * args.log_interval * args.world_size)
                reduce_loss_hardneg = reduce_loss_hardneg.item() / (args.gradient_accumulation_steps * args.log_interval * args.world_size)
                reduce_loss_rd = reduce_loss_rd.item() / (args.gradient_accumulation_steps * args.log_interval * args.world_size)
                reduce_loss_rd2 = reduce_loss_rd2.item() / (args.gradient_accumulation_steps * args.log_interval * args.world_size)
                reduce_loss_feat = reduce_loss_feat.item() / (args.gradient_accumulation_steps * args.log_interval * args.world_size)

                if args.global_rank == 0:
                    train_log_dict = {}
                    train_log_dict['loss_overall'] = reduce_loss
                    train_log_dict = {'loss_inbatch':reduce_loss_in_batch}
                    train_log_dict['loss_hardneg'] = reduce_loss_hardneg
                    train_log_dict['loss_rd'] = reduce_loss_rd
                    train_log_dict['loss_rd2'] = reduce_loss_rd2
                    train_log_dict['loss_feat'] = reduce_loss_feat
                    write_tensorboard(summary_writer, train_log_dict, global_step)

                reduce_loss = 0
                reduce_loss_hardneg = 0
                reduce_loss_rd = 0
                reduce_loss_rd2 = 0
                reduce_loss_feat = 0
                reduce_loss_in_batch = 0

            if global_step % args.eval_interval == 0:
                model_engine.eval()
                batch_iterator_eval = tqdm(val_dataloader,
                                        disable=(args.global_rank!=0),
                                        mininterval=0)
                
                with torch.no_grad():
                    for step, batch in enumerate(batch_iterator_eval):
                        sentence_a, sentence_b, _, _, _, task_id = batch
                        sentence_all = sentence_a + sentence_b
                        bs = dataset_id.size(0)

                        key = 'global_rank' + str(args.global_rank)

                        inputs_all = model.tokenizer(sentence_all, padding='max_length', max_length=args.max_seq_length, truncation=True, return_tensors='pt')
            
                        inputs_all = inputs_all.to(device)
                        task_id = task_id.to(device)
                        logits_student_in_batch_eval, _, _ = model_engine(inputs_all, task_id, 'eval')

                        loss_in_batch_dict_eval = cal_loss_in_batch(args, logits_student_in_batch_eval, args.temperature_in_batch, criterion)
                    
                        loss_batch_eval = loss_in_batch.detach()
                        if args.verbose:
                            batch_iterator_eval.set_description(
                                f"Epoch: {current_epoch + 1}/{args.num_epochs}, Batch:{step}/{len(val_dataloader)}, Loss: {loss_batch_eval:9.4f}")


                        reduce_loss_eval += loss_batch_eval
                        
                    dist.all_reduce(reduce_loss_eval, op=dist.ReduceOp.SUM)
                    reduce_loss_eval = reduce_loss_eval.item() / (val_loader_size * args.world_size)
                    
                    if args.global_rank == 0:
                        eval_log_dict = {'loss_eval':reduce_loss_eval}
                        write_tensorboard(summary_writer, eval_log_dict, global_step)

                save_flag = False

                if stop >= args.patience:
                    break

                if reduce_loss_eval <= min_reduce_loss_eval:
                    min_reduce_loss_eval = reduce_loss_eval
                    best_epoch = current_epoch
                    best_step = global_step
                    stop = 0
                    
                    path = args.output_dir
                    start_name = 'checkpoint'
                    current_step_num = global_step
                    max_save_num = 2
                    if args.global_rank == 0:
                        print('removing')
                        try:
                            remove_earlier_ckpt(path, start_name, current_step_num, max_save_num)  
                        except:
                            print('No ckpt to remove.')
                else:
                    stop += 1

                if stop < args.num_ckpt:
                    save_flag = True


                if save_flag:
                    output_dir_current = os.path.join(args.output_dir, "checkpoint-{}-epoch-{}-{}".format(global_step, current_epoch+1, args.mark))
                    client_sd = dict()

                    save_model(model_engine, output_dir_current, client_state=client_sd)

                reduce_loss_eval = 0
                model_engine.train()


if __name__ == '__main__':
    main()
