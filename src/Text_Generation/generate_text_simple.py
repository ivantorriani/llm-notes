#=========================
'''
file: generate_text_simple.py
purpose: print the outputs of the model to track training progress. 
note: need to sort out generate_text_simple, has issues with the config dependencies 
that i'll deal with later. should put on the priority queue. 
'''
#=========================

#imports=========================
import torch 
#imports=========================

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:] #get only the context length tokens


        with torch.no_grad():
            logits = model(idx_cond) #pass it through the model.

        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx