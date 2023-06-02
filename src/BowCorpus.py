# Made by Heikki Kupiainen 2023

import gensim
from gensim import corpora
from gensim.utils import simple_preprocess

import TokenizedList
import PhraseList

class BowCorpus:
  def __init__(self, dictionary, corpus):
    self.dictionary = dictionary
    self.corpus = corpus
    
def build_from_phrase_list(phrase_list: list[str]):
  tokenized_list = TokenizedList.build_from_phrase_list(phrase_list)
  dictionary = corpora.Dictionary()
  corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_list]
  return BowCorpus(dictionary, corpus)

def build_from_sample_phrase_list():
  phrase_list = PhraseList.get_sample_phrases()
  return build_from_phrase_list(phrase_list)

def build_from_tokenized_list(tokenized_list):
  dictionary = corpora.Dictionary()
  corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_list]
  return BowCorpus(dictionary, corpus)
  
def get_word_counts(bow_corpus: BowCorpus):
  return [[(bow_corpus.dictionary[id], count) for id, count in line] for line in bow_corpus.corpus]
    
def count_words_in_sample_corpus():
  bow_corpus = build_from_sample_phrase_list()
  word_counts = get_word_counts(bow_corpus)
  print(word_counts)