import os
import sys
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

names = sys.argv[1:]

### TODO: read data from ([name]/input.txt for name in names)
### TODO: combine multiple books into one single data file
data = ''
for name in names:
    with open(os.path.join(name, "input.txt"), "r", encoding="utf-8") as f:
        data += f.read()
###

### TODO: split data for train(0.9) and valid (0.1)
train_data = data[:int(len(data)*0.9)]
val_data = data[int(len(data)*0.9):]
###

### TODO: tokenize raw data with tiktoken encoder
### TODO: transform to numpy array
train_ids = np.array(enc.encode_ordinary(train_data), dtype=np.uint16)
val_ids = np.array(enc.encode_ordinary(val_data), dtype=np.uint16)
###

# save numpy array to file processed_pretrain/train.bin and processed_pretrain/val.bin
train_ids.tofile(os.path.join("processed_pretrain", "train.bin"))
val_ids.tofile(os.path.join("processed_pretrain", 'val.bin'))
