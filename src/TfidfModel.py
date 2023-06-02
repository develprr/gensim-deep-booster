# Made by Heikki Kupiainen 2023

import BWCorpus

from gensim import models
from gensim import corpora

import numpy as np

class TfidfModel(object):
  
  def __init__(self,bw_corpus: BWCorpus.Model):
    self.corpus = bw_corpus
    self.model = models.TfidfModel(bw_corpus.corpus, smartirs='ntc')

    
  def print(self):
    for doc in self.corpus:
      print([[[id], np.around(freq, decimals=2)] for id, freq in doc])
      
def build_from_bw_corpus(bw_corpus: BWCorpus.Model):
  return TfidfModel(bw_corpus)


def get_dictionary(model: TfidfModel):
    corpus = model.corpus
    return BWCorpus.get_dictionary(corpus);
    

def build_sample() -> TfidfModel:
  bw_corpus = BWCorpus.build_from_sample_phrase_list()
  return build_from_bw_corpus(bw_corpus)

def test_build_sample():
  model = build_sample()
  assert(type(model)) == TfidfModel

def test_get_dictionary():
  model = build_sample()
  dictionary = get_dictionary(model) 
  assert(type(dictionary)) == corpora.Dictionary

def test_build_corpus():
  bw_corpus = BWCorpus.build_from_sample_phrase_list()
  print(type(bw_corpus.corpus))
  
  assert(type(bw_corpus)) == BWCorpus.Model
  tfidf_model = build_from_bw_corpus(bw_corpus)
  
  assert(type(tfidf_model)) == TfidfModel
  assert(type(tfidf_model.corpus)) == BWCorpus.Model
  assert(type(tfidf_model.model)) == models.tfidfmodel.TfidfModel