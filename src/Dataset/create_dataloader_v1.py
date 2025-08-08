#=========================
'''
file: create_dataloader_v1.py
purpose: initialize the dataloader to be fed into the model
'''
#=========================

#imports=========================
import torch
from torch.utils.data import  DataLoader
from src.Dataset.DatasetMaker import GPTDatasetV1
from src.text_loaders.read_text import readtxt
from src.Tokenizers.SimpleTokenizers import byte_tokenizer
#imports=========================



def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = byte_tokenizer

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

#dataloader_1 = (txt, byte tokenizer, 1, 4, 1, shuffle=True)
#dataloader_large = (txt,  8, 4, 4, True ) # for larger, more parallel processes 