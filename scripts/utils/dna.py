"""Helpers for working with DNA/RNA data"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class DNADataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# vocabulary map
dna_vocab = {
    "A":0, 
    "C":1,
    "G":2,
    "T":3,
}
dna_with_padding_vocab = {
    "A":0, 
    "C":1,
    "G":2,
    "T":3,
    "N":4
}

def one_hot(line, max_len, charmap):
    chars = line.strip()
    I = np.eye(len(charmap))
    try:
        base = [I[charmap[c]] for c in chars]
        if len(chars) < max_len:
            extra = []
            if "*" in charmap: extra = [I[charmap["*"]]] * (max_len - len(chars))
        else:
            extra = []
        arr = np.array(base + extra)
    except:
        raise Exception("Unable to process line: {}".format(chars))
    return np.expand_dims(arr, 0)

def load(data_loc, seq_len, vocab="ATGC", random_seed=42):
    charmap = dna_vocab if vocab == "ATGC" else dna_with_padding_vocab

    data = {}
    for dataType in ['train', 'val']:
        with open(os.path.join(data_loc, f'{dataType}.txt')) as f:
            lines = f.readlines()

        lines = [one_hot(line, seq_len, charmap) for line in lines]
        lines = np.vstack(lines)
        data[dataType] = lines

    return data

def get_vocab(vocab):
    return dna_vocab if vocab == "ATGC" else dna_with_padding_vocab