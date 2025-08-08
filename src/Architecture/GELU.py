#=========================
'''
file: GELU.py
purpose: After going through multi-head attention and normalization, 
we need to start extracting relationships in the data in the dense layers.
We use a gelu function. 

'''
#=========================

#imports=========================
import torch
import torch.nn as nn
#imports=========================

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))