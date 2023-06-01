# Made by Heikki Kupiainen 2023

import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from smart_open import smart_open
from dotenv import load_dotenv
import os

import SourceTextDirectory

load_dotenv()

DEFAULT_TEXT_SOURCE_FILE = os.getenv('DEFAULT_TEXT_SOURCE_FILE')    
DEFAULT_TEXT_SOURCE_DIR = os.getenv('DEFAULT_TEST__SOURCE_DIR')    

def build_from_source_file(filepath: str):
  return corpora.Dictionary(simple_preprocess(line, deacc=True) for line in open(filepath, encoding='utf-8'))

def build_from_default_source_file(): 
  return build_from_source_file(DEFAULT_TEXT_SOURCE_FILE)
  
def build_from_source_directory(dirpath: str): 
  return SourceTextDirectory.read_files(dirpath)
    
def build_from_default_source_directory():
  return build_from_source_directory(DEFAULT_TEXT_SOURCE_DIR)