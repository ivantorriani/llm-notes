#=========================
'''
file: SelfAttentionv1.py
purpose: Initiate a class to implement self attention mechanisms.  
'''
#=========================

#imports=========================
import torch
import torch.nn as nn
#imports=========================

'''
purpose: generalize attention mechanisms
'''

class SelfAttentionv1(nn.module):
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
