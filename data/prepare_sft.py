### TODO: prepare SFT data similar to `prepare.py`

'''
进行SFT训练的流程与预训练阶段几乎相同，但在数据预处理与损失函数计算位置的设置上略有不同：
1.预处理：对于单条SFT数据，我们将问与答两部分合并，截断或填充至固定长度（block size），来输入模型。本作业需要同学们参照预训练阶段，完成这一预处理的实现。
    1).利用文本编码器对question和answer分别编码，同时进行适当的拼接
    2).通过适当标识区分question和answer
    3).训练集和验证集划分。
    4).处理不同长度的数据
2.损失函数：data_utils.py
'''

import os
import sys
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

names = sys.argv[1:]

# read data from ([name]/sft.txt for name in names)
# add in special str "q" and "a" to distinguish question and answer
# tokenize raw data with tiktoken encoder
train_tokens = []
val_tokens = []
for name in names:
    with open(os.path.join(name, "sft.txt"), "r", encoding="utf-8") as f:
        data_str = f.read()
    data_list = data_str.split('\n')
    for data_ in data_list[:int(0.9*len(data_list))]:
        train_data = ('q' + eval(data_)['Question'].strip() + 'a' + eval(data_)['Answer']).strip()
        train_tokens += (enc.encode_ordinary(train_data) + [enc.eot_token])
    for data_ in data_list[int(0.9*len(data_list)):]:
        val_data = ('q' + eval(data_)['Question'].strip() + 'a' + eval(data_)['Answer']).strip()
        val_tokens += (enc.encode_ordinary(val_data) + [enc.eot_token])
        
# transform to numpy array
train_ids = np.array(train_tokens, dtype=np.uint16)
val_ids = np.array(val_tokens, dtype=np.uint16)

# save numpy array to file processed_sft/train.bin and processed_sft/val.bin
train_ids.tofile(os.path.join("processed_sft", "train.bin"))
val_ids.tofile(os.path.join("processed_sft", "val.bin"))


###