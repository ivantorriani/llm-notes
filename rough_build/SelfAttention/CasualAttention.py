#=========================
'''
file: CasualAttention.py
purpose: Implement a casual attention class (self attention + masking )
'''
#=========================

#imports=========================
import torch
import torch.nn as nn
#imports=========================

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        # also need to initialize mask and dropout
        self.dropout = dropout 
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
