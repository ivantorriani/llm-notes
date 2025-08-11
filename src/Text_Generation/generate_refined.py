#=========================
'''
file: generate_refined.py
purpose: Text generation with temperature and top k scaling, 
leading to much less 
'''
#=========================

#imports=========================
import torch 
#imports=========================

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # get last logits
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        #top k sampling
        if top_k is not None:
            
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # temp scaling
        if temperature > 0.0:
            logits = logits / temperature

            # softmax
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # multinomial to get various probabilties
            idx_next = torch.multinomial(probs, num_samples=1) 

        #if not just apply regular argmax
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  
            break

        # generate text gpt style
        idx = torch.cat((idx, idx_next), dim=1)  

    return idx