# Made by Heikki Kupiainen 2023

import gensim
from gensim import corpora
from gensim.utils import simple_preprocess

def get_sample_phrases():
  return [
    "Jupiter is the biggest planet",
    "Mercury is the smallest planet"
  ]
  
def tokenize(phrases: list[str]):
  return [simple_preprocess(doc) for doc in phrases]

def tokenize_sample_phrases():
  return tokenize(get_sample_phrases())
