#=========================
'''
file: evaluate_xpuryv2.py
purpose: Evaluate the performance of a newly trained model by printing some text
note: python3 -m scripts.evaluate.evaluate_xpuryv2.py
'''
#=========================

#imports=========================
import torch
import json 
import tiktoken
from src.Text_Generation.generate_refined import generate 
from src.Architecture.Model import GPTModel

#imports=========================

#import in the config
with open("src/Architecture/MODEL_CONFIGS.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

#initialize model
model = GPTModel(cfg["XPERYV2_CONFIG"])
model.load_state_dict(torch.load("checkpoints/exuperyv2.pth"))

#initialize tokenizer
tokenizer=tiktoken.get_encoding("gpt2")

    
if __name__ == "__main__":
    start_words = "Hello, I am"
    tokenized_words = tokenizer.encode(start_words)
    idx = torch.tensor([tokenized_words])
    model_test = generate(
        model, 
        idx=idx,
        max_new_tokens=20,
        context_size=50,
        temperature=0.7,
        top_k=4,
        eos_id=None
    )
    print(tokenizer.decode(model_test[0].tolist()))

