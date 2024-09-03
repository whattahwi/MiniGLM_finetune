import os

import torch
import numpy as np
import tiktoken
import random

enc = tiktoken.get_encoding("gpt2")

train_data = None
val_data = None

def init_data_pretrain(dataset):
    global train_data, val_data
    
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def init_data_sft(dataset):
    global train_data, val_data
    
    ### TODO: 读取+初始化sft数据
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ###

def get_batch_pretrain(split, batch_size, block_size, device):
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    # 从数据中随机选择batch_size个索引，这些索引将用于提取批次(batch)的数据
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    loss_mask = torch.ones_like(x, dtype=torch.float64)
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y, loss_mask

x_list = []
y_list = []
loss_mask_list = []
    
def get_batch_sft(split, block_size, device, batch_size=10): 
    ### TODO: 获取sft数据的批次（batch）+ 构建损失函数掩码（loss_mask）
    global train_data, val_data
    data = train_data if split == 'train' else val_data
    
    global x_list, y_list, loss_mask_list
    
    if len(x_list) == 0: # just do this work for 1 time
        target_q = np.array(enc.encode_ordinary('q'), dtype=np.uint16)
        target_a = np.array(enc.encode_ordinary('a'), dtype=np.uint16)
        data = np.split(data, np.where(data==target_q[0])[0])
        for data_ in data[1:]:
            x_list.append(data_[:-1])
            y_list.append(data_[1:])
            temp = np.split(x_list[-1], np.where(x_list[-1]==target_a[0])[0])
            loss_mask_list.append(np.append(np.zeros_like(temp[0]),np.ones_like(temp[1])))

    ix = random.sample(range(len(x_list)), batch_size) # 选取哪些问答数据
    
    max_length = max([len(x_list[i]) for i in ix])
    for i in ix:
        x_list[i] = np.append(x_list[i], np.zeros(max_length-len(x_list[i])))
        y_list[i] = np.append(y_list[i], np.zeros(max_length-len(y_list[i])))
        loss_mask_list[i] = np.append(loss_mask_list[i], np.zeros(max_length-len(loss_mask_list[i])))
        if block_size < max_length:
            x_list[i] = x_list[i][:block_size]
            y_list[i] = y_list[i][:block_size]
            loss_mask_list[i] = loss_mask_list[i][:block_size]
    
    x = torch.stack([torch.from_numpy((x_list[i]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((y_list[i]).astype(np.int64)) for i in ix])
    loss_mask = torch.stack([torch.from_numpy((loss_mask_list[i]).astype(np.int64)) for i in ix])
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    ###
    
    return x, y, loss_mask