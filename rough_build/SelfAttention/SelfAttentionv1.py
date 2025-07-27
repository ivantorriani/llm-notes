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
class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    '''
    purpose: given an input tensor (of sentence of length whatever)
    return the context vector. 
    '''
    def forward(self, x): #x is an entire input tensor
        keys = x @ self.W_key  
        queries = x @ self.W_query
        values = x @ self.W_value #access the first by values[0], same for above
        
        attn_scores = queries @ keys.T 
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec