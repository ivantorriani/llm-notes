#=========================
'''
processing text here
'''
#=========================

import os
import re

#just get the damn thing absolutely
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../../text-data/news_story.txt")

with open(file_path, 'r', encoding="utf-8") as f:
    raw_text = f.read()

preprocess = re.split(r'([,.:;?_!"()\s]|--)', raw_text)
preprocess = [item.strip() for item in preprocess if item.strip()]
print(preprocess[:50])
