#=========================
'''
file: Simple-Tokenizers.py
purpose: present two simple methods for decoding and encoding text
'''
#=========================
# python3 -m rough_build.tokenizers.simple_tokenizers

#imports==============================================================
import re
import tiktoken
from rough_build.text_loaders.read_text import readtxt

#process text===========================================================================

raw_text = readtxt("news_story.txt")

preprocess_pass_one = re.split(r'([,.:;?_!"()\s]|--)', raw_text)
preprocess_pass_two = [item.strip() for item in preprocess_pass_one if item.strip()]

#vocabulary===========================================================================

alphabetical_words = sorted(set(preprocess_pass_two))
alphabetical_words.extend(["<|endoftext|>", "<|unk|>"])
vocabulary = {token:interger for interger, token in enumerate(alphabetical_words)}

# tokenizers (not handle unknowns, handles unknowns, premade) ==================================================

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\s]|--)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

# set tokenizers ==================================================

tokenizer = SimpleTokenizerV1(vocabulary)
tokenizer_2 = SimpleTokenizerV2(vocabulary)
byte_tokenizer = tiktoken.get_encoding("gpt2")
encoded_text = byte_tokenizer.encode(raw_text, allowed_special={'<|endoftext|>'})

# set tokenizers ==================================================