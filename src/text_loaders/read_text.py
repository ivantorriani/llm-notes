#=========================
'''
file: read-text.py
purpose: read txt files and return raw strings
'''
#=========================

def readtxt(path:str) -> str:    
    with open(path, 'r', encoding="utf-8") as f:
        raw_text = f.read()
        return raw_text