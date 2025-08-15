#=========================
'''
file: download_weights.py
purpose: Download weights from internet. 
note: python3 -m scripts.downloads.download_weights

'''
#=========================

#imports=========================
import urllib.request 
from gpt_download import download_and_load_gpt2
#imports=========================

url = (
 "https://raw.githubusercontent.com/rasbt/"
 "LLMs-from-scratch/main/ch05/"
 "01_main-chapter-code/gpt_download.py"
)

if __name__ == "__main__":
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)

    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
