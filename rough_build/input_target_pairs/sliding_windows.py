#=========================
'''
file: sliding_windows.py
purpose: create text pair encodings for next word predictions, or more generally just demonstrate functionality
run: python3 -m rough_build.input_target_pairs.sliding_windows
'''
#=========================


#imports=========================
from rough_build.text_loaders.read_text import readtxt
from rough_build.tokenizers.SimpleTokenizers import byte_tokenizer
#imports=========================


#main=========================

#load and encode text
raw_text = readtxt("news_story.txt")
encoded_text = byte_tokenizer.encode(raw_text, allowed_special={'<|endoftext|>'})
encoded_sample = encoded_text[:100]

context_size = 5

for i in range(1, context_size+1):
    context = encoded_sample[:i]
    desired = encoded_sample[i]
    print(context, " ----->", desired, '\n')
    print(byte_tokenizer.decode(context),  " ----->", byte_tokenizer.decode([desired]), '\n', '===================', '\n')