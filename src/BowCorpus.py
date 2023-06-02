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

def build_from_file(filepath: str):
  corpus = []
  dictionary = corpora.Dictionary()
  for line in smart_open(filepath, encoding='utf8'):
    tokenized_list = simple_preprocess(line, deacc=True)
    bow = dictionary.doc2bow(tokenized_list, allow_update=True)
    corpus.append(bow)
  return BowCorpus(dictionary, corpus)

def build_from_sample_file():
  print(SAMPLE_TEXT_SOURCE_FILE)

  corpus = build_from_file(SAMPLE_TEXT_SOURCE_FILE)
  print(corpus)