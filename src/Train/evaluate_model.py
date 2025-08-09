#=========================
'''
file: evaluate_model.py
purpose: Print out the training losses and evaluation losses
'''
#=========================

#imports=========================
import torch 
import torch.nn as nn
from src.Analysis.cross_entropy import calc_loss_loader 
#imports=========================

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
