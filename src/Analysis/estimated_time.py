#=========================
'''
file: estimated_time.py
purpose: Estimate roughly how long a process
will take. 
note:
max_length = 256, stride =128
'''
#=========================

#imports=========================
import torch
import json 
import tiktoken
import torch.nn as nn 
from src.text_loaders.read_text import readtxt
from src.Dataset.create_dataloader_v1 import create_dataloader_v1
#imports=========================

tokenizer = tiktoken.get_encoding("gpt2")

#get configurations
with open("src/Architecture/MODEL_CONFIGS.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)


#calculate steps harshly (more than actual). 
def calculate_steps_critical(tokens, sequence_length, batch_size): 
    return ((tokens) / (sequence_length * batch_size))

if __name__ == "__main__":
    #read text according to input
    text_path = input("Path to data: ")
    raw_text = readtxt(text_path)

    #get epoch amount

    #get total tokens
    total_tokens = len(tokenizer.encode(raw_text))

    #get batch size
    total_batch = 4

    # get sequence (context) length
    sequence_length = cfg["XPERYV2_CONFIG"]["context_length"]

    print("Critical Steps per Epoch: ",calculate_steps_critical(total_tokens, sequence_length, total_batch))
