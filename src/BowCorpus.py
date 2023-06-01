
# Made by Heikki Kupiainen 2023

import gensim
from gensim import corpora

import TokenizedList
import PhraseList

def create_from_phrase_list(phrase_list: list[str]):
  tokenized_list = TokenizedList.build_from_phrase_list(phrase_list)
  dictionary = corpora.Dictionary()
  return [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_list]

def create_from_sample_phrase_list():
  phrase_list = PhraseList.get_sample_phrases()
  return create_from_phrase_list(phrase_list)

