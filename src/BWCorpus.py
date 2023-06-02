# Made by Heikki Kupiainen 2023

import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from smart_open import smart_open
import nltk
import os
from dotenv import load_dotenv

nltk.download('stopwords')  # run once
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import TokenizedList
import PhraseList

load_dotenv()

SAMPLE_TEXT_SOURCE_FILE = os.getenv('SAMPLE_TEXT_SOURCE_FILE')    
CORPUS_SERIALIZATION_DIR = os.getenv('CORPUS_SERIALIZATION_DIR')

class BWCorpus(object):

  def __init__(self, dictionary, corpus):
    self.dictionary = dictionary
    self.corpus = corpus
    
  def __repr__(self):
      return f"<BWCorpus corpus:{self.corpus} dictionary:{self.dictionary} >"
    
def get_dictionary(corpus: BWCorpus):
  return corpus.dictionary

def get_corpus(corpus: BWCorpus):
  return corpus.corpus

def build_from_sample_phrase_list() -> BWCorpus:
  phrase_list = PhraseList.get_sample_phrases()
  return build_from_phrase_list(phrase_list)

def build_from_phrase_list(phrase_list: list[str]) -> BWCorpus:
  tokenized_list = TokenizedList.build_from_phrase_list(phrase_list)
  dictionary = corpora.Dictionary()
  corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_list]
  return BWCorpus(dictionary, corpus)

def test_build_from_phrase_list():
  bw_corpus = build_from_sample_phrase_list()
  assert(type(bw_corpus)) == BWCorpus
  dictionary = get_dictionary(bw_corpus)
  assert(type(dictionary)) == corpora.Dictionary
  corpus = get_corpus(bw_corpus)
  assert(type(corpus)) == list