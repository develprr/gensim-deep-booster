# Made by Heikki Kupiainen 2023

from BWCorpus import BWCorpus
from gensim import models
import numpy as np

class TfidfModel(object):
  
  def __init__(self,corpus):
    self.corpus = corpus
    self.model = models.TfidfModel(corpus, smartirs='ntc')

  @staticmethod
  def build_from_bw_corpus(bw_corpus: BWCorpus):
    corpus = bw_corpus.corpus
    return TfidfModel(corpus)
  
def test_build_corpus():
  bw_corpus = BWCorpus.build_from_sample_phrase_list()
  print(type(bw_corpus.corpus))
  
  assert(type(bw_corpus)) == BWCorpus
  tfidf_model = TfidfModel.build_from_bw_corpus(bw_corpus)
  
  assert(type(tfidf_model)) == TfidfModel
  assert(type(tfidf_model.corpus)) == list
  assert(type(tfidf_model.model)) == models.tfidfmodel.TfidfModel