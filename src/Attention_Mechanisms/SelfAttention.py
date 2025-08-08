#=========================
'''
file: SelfAttentionv1.py
purpose: Initiate a class to implement self attention mechanisms.  
note: if buggy results occur in the future, consider implementing
the book's algorithms. it might be a line or two of my own
problems causing mishaps, but it looks mostly fine. 
'''
#=========================

#imports=========================
import torch
import torch.nn as nn
#imports=========================

'''
purpose: generalize attention mechanisms
'''

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        '''
        nn.Linear asserts the weight values, but also can be used to calcluate the queries 
        values
        '''
    def forward(self, x):# where x is the tensor
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        unnormalized_attn_scores = queries @ keys.tranpose(0,1)
        normalized_attn_scores = torch.softmax(((unnormalized_attn_scores) / (keys.shape[-1])**(.5)), dim=-1)
        context_vector = normalized_attn_scores @ values 

        return context_vector

