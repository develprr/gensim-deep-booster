# (C) by Heikki Kupiainen 2023    

import gensim
from gensim import corpora
from api import Api

class Bigram(object):
  
  def __init__(self, dataset):
    self.dataset = dataset
    self.phrases = gensim.models.phrases.Phrases(dataset, min_count=3, threshold=10)

  def build_from_text8():
    dataset = Api.load_text8_dataset()
    dct = corpora.Dictionary(dataset)
    [dct.doc2bow(line) for line in dataset]
    return Bigram(dataset)
    
def test_build_from_text8():
  bigram = Bigram.build_from_text8()
  dataset = bigram.dataset
  phrases = bigram.phrases
  assert(type(phrases)) == gensim.models.phrases.Phrases