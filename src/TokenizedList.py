# Made by Heikki Kupiainen 2023

# TokenizedList module provides tools to convert phrase lists
# into tokenized lists

import gensim
import PhraseList
from gensim import corpora
from gensim.utils import simple_preprocess

def build_from_phrase_list(phrase_list: list[str]):
  return [simple_preprocess(doc) for doc in phrase_list]

def build_from_sample_phrase_list():
  phrase_list = PhraseList.get_sample_phrases()
  return build_from_phrase_list(phrase_list)  
