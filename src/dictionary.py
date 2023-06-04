# (C) by Heikki Kupiainen 2023    


import gensim
from gensim import corpora

from api import Api

class Dictionary(object): 
  
  def __init__(self, corpora_dictionary: corpora.Dictionary):
    self.corpora_dictionary = corpora_dictionary

  def __repr__(self):
      return f"<Dictionary corpora_dictionary:{self.corpora_dictionary}>"
      
  @staticmethod
  def build_from_text8():
    dataset = Api.load_tex8_dataset()
    corpora_dictionary = corpora.Dictionary(dataset)
    return Dictionary(corpora_dictionary)