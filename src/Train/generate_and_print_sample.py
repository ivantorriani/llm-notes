#=========================
'''
file: generate_and_print_sample.py
purpose: print the outputs of the model to track training progress. 
note: need to sort out generate_text_simple, has issues with the config dependencies 
that i'll deal with later. should put on the priority queue. 
'''
#=========================

#imports=========================
import torch 
import torch.nn as nn
from src.Text_Generation.tok_text_transitions import text_to_token_ids, token_ids_to_text
from src.Text_Generation.generate_text_simple import generate_text_simple

#imports=========================
'''
model.eval() # no dropouts because we're not traiing

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    contex
'''
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()