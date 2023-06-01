# Made by Heikki Kupiainen 2023

import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from smart_open import smart_open
from dotenv import load_dotenv
import os

load_dotenv()

TEXT_SOURCE_DIR = os.getenv('TEXT_SOURCE_DIR')    

def read_files(directory_path: str):
  for fname in os.listdir(directory_path):
    for line in open(os.path.join(directory_path, fname), encoding='utf8'):
      yield simple_preprocess(line)

def read_files_from_sample_directory():
  return read_files(TEXT_SOURCE_DIR)