#=========================
'''
file: train_v1.py
purpose: Simple trainig sequence on extremely small dataset (the verdict). Just want
to see how it works
'''
#=========================

#imports=========================
import json 
from src.Analysis.cross_entropy import calc_loss_batch, calc_loss_loader
from src.Dataset.DatasetMaker import GPTDatasetV1
from src.Dataset.create_dataloader_v1 import create_dataloader_v1
from src.text_loaders.read_text import readtxt

#imports=========================

#open json file for configs

with open("../src/Architecture/MODEL_CONFIGS.json", "r", encoding="utf-8") as f:
    config = json.load(f)

print(config)


