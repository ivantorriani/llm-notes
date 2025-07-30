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

class CasualAttention(nn.module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        # also need to initialize mask and dropout
        self.dropout = dropout 
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.context_length = context_length
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries, keys, values = self.W_query(x), self.W_keys(x), self.W_values(x)
        unnormalized_attn_scores = queries @ (keys.transpose(1,2))

        #mas/dropout phase
        unnormalized_attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        normalized_attn_scores = torch.softmax(unnormalized_attn_scores / keys.shape[-1]**0.5, dim=1) 
        normalized_attn_scores = self.dropout(normalized_attn_scores)

        #return context vector
        
        return normalized_attn_scores @ values


        

