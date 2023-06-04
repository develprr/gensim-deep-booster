# (C) Heikki Kupiainen 2023    

import BWCorpus

from gensim import models
from gensim import corpora

import numpy as np

class TfIdf(object):
  
  def __init__(self,bw_corpus: BWCorpus.Model):
    self.bw_corpus = bw_corpus
    self.model = models.TfidfModel(bw_corpus.corpus, smartirs='ntc')

  def __repr__(self):
      return f"<TfiIdf bw_corpus:{self.bw_corpus} model:{self.model} >"
  
  def get_bw_corpus(self):
    return self.bw_corpus
  
  def get_corpora_corpus(self):
    bw_corpus = self.get_bw_corpus()
    return BWCorpus.get_corpora_corpus(bw_corpus)

  def get_dictionary(self):
    return BWCorpus.get_dictionary(self.bw_corpus);    
  
  def print(self):
    for doc in self.get_corpora_corpus():
      print([[self.get_dictionary()[id], np.around(freq, decimals=2)] for id, freq in doc])
    
def build_sample() -> TfIdf:
  bw_corpus = BWCorpus.build_sample()
  return TfIdf(bw_corpus)

def test_build_sample():
  model = build_sample()
  assert(type(model)) == TfIdf

def test_get_dictionary():
  tf_idf = build_sample()
  dictionary = tf_idf.get_dictionary() 
  assert(type(dictionary)) == corpora.Dictionary

def test_get_bw_corpus():
  tf_idf = build_sample()
  print(tf_idf)
  bw_corpus = tf_idf.get_bw_corpus()
  assert(type(bw_corpus)) == BWCorpus.Model
  
def test_print():
  tf_idf = build_sample()
  tf_idf.print()

def test_get_corpora_corpus():
  tf_idf = build_sample()
  ccorpus = tf_idf.get_corpora_corpus()
  assert(type(ccorpus)) == list

def test_build_corpus():
  bw_corpus = BWCorpus.build_sample()
  print(type(bw_corpus.corpus))
  assert(type(bw_corpus)) == BWCorpus.Model
  tfidf_model = TfIdf(bw_corpus)
  assert(type(tfidf_model)) == TfIdf
  assert(type(tfidf_model.bw_corpus)) == BWCorpus.Model
  assert(type(tfidf_model.model)) == models.tfidfmodel.TfidfModel