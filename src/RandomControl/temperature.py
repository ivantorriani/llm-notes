#=========================
'''
file: temperature.py
purpose: Apply a function to use temperature in order to diversify the text generation.
'''
#=========================

#imports=========================
import torch
#imports=========================

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)