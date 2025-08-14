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
from src.Text_Generation.generate_refined import generate 
from src.Architecture.Model import GPTModel
#imports=========================

#import in the config
with open("src/Architecture/MODEL_CONFIGS.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

model = GPTModel(cfg["GPT_CONFIG_124M"])
model.load_state_dict(torch.load("checkpoints/exuperyv2.pth"))

    
if __name__ == "__main__":
    model.eval()

