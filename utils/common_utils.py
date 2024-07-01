import os
import random
import pathlib
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
from loguru import logger
import shutil
from torch.utils.tensorboard import SummaryWriter
import pickle
import linecache
import tracemalloc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model_engine, ckpt_dir, client_state):
    model_engine.save_checkpoint(ckpt_dir, client_state=client_state, exclude_frozen_parameters=True)

def remove_earlier_ckpt(path, start_name, current_step_num, max_save_num):

    filenames=os.listdir(path)
    ckpts = [dir_name for dir_name in filenames if dir_name.startswith(start_name) and int(dir_name.split('-')[1])<=current_step_num]
    
    current_ckpt_num = len(ckpts)
    for dir_name in filenames:
        if dir_name.startswith(start_name) and int(dir_name.split('-')[1]) <= current_step_num and current_ckpt_num > (max_save_num-1):
            shutil.rmtree(os.path.join(path, dir_name))


def makedirs(path):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return path

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def write_pickle(obj, path:str):
    if not os.path.exists(path):
        makedirs(path)
    with open(path, "wb") as f:
        return pickle.dump(obj, f)

def write_tensorboard(summary_writer, log_dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(f'{key}', value, completed_steps)

def cos_sim(a, b):

    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

