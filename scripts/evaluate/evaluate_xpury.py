#=========================
'''
file: evaluate_xpuryv2.py
purpose: Evaluate the performance of a newly trained model by printing some text
note: python3 -m scripts.evaluate.evaluate_xpury

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
model = GPTModel(cfg["GPT_CONFIG_124M"])
#model.load_state_dict(torch.load("checkpoints/exuperyv2.pth"))

#initialize tokenizer
tokenizer=tiktoken.get_encoding("gpt2")

    
if __name__ == "__main__":
    model_path = input("Enter model name: ")
    model.load_state_dict(torch.load("checkpoints/" + model_path), map_location=torch.device('cpu'))
    start_words = "Are we capable of more than our abstract ceilings?"
    tokenized_words = tokenizer.encode(start_words)
    idx = torch.tensor([tokenized_words])
    model.eval()
    model_test = generate(
        model, 
        idx=idx,
        max_new_tokens=100,
        context_size=20,
        temperature=0.7,
        top_k=40,
        eos_id=None
    )
    decoded_text = tokenizer.decode(model_test[0].tolist())
    clean_text = "\n".join([line for line in decoded_text.splitlines() if line.strip() != ""])
    print(clean_text)
