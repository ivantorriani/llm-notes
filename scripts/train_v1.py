#=========================
'''
file: train_v1.py
purpose: Simple trainig sequence on extremely small dataset (the verdict). Just want
to see how it works
note: python3 -m scripts.train_v1

'''
#=========================

#imports=========================
import json 
import torch
import tiktoken
from src.Train.train_model_simple import train_model_simple
from src.Architecture.Model import GPTModel
from src.Dataset.create_dataloader_v1 import create_dataloader_v1
from src.text_loaders.read_text import readtxt

#imports=========================


#Read text
raw_text = readtxt("data/the-verdict.txt")

#Open configs
with open("src/Architecture/MODEL_CONFIGS.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

config = cfg["GPT_CONFIG_124M"]

#Initialize training data
train_ratio = 0.90
split_idx = int(train_ratio * len(raw_text))
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]

#Initialize model
model = GPTModel(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=.0003,weight_decay=0.1
)

#Initialize data loaders

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=config["context_length"],
    stride=config["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=config["context_length"],
    stride=config["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

#Set up training environment
num_epochs = 10

train_loss, val_loss, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, 
    eval_freq=5, eval_iter = 5, start_context="Why do the birds sing? ", tokenizer=tiktoken.get_encoding("gpt2")
)

print(raw_text)


