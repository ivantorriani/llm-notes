#=========================
'''
file: cross_entropy.py
purpose: Calculate the accuracy of the model's predictions to the targets. 
note: functions included:
calc_loss_batch
'''
#=========================

#imports=========================
import torch 
import torch.nn as nn 

#imports=========================

'''purpose

'''
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss 