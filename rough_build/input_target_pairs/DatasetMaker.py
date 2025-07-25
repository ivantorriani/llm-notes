#=========================
'''
file: DatasetMaker.py
purpose: create text pair encodings for next word predictions with a defined class functionality
run: python3 -m rough_build.input_target_pairs.DatasetMaker
'''
#=========================

#imports=========================
import torch
from torch.utils.data import Dataset, DataLoader
from rough_build.text_loaders.read_text import readtxt
from rough_build.tokenizers.simple_tokenizers import byte_tokenizer
#imports=========================

class GPTDatasetV1(Dataset):
    '''
    input: raw text, tokenizer selection, length of 
    '''
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]