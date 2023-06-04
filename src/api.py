# (C) by Heikki Kupiainen 2023    

import gensim.downloader as downloader

class Api(object):
  
  @staticmethod  
  def load_text8_body():
    return downloader.load("text8")

  @staticmethod  
  def load_tex8_dataset():
    body = Api.load_text8_body()
    dataset = [wd for wd in body]
    return dataset
    

def info_gigaworld():
  print(downloader.info('glove-wiki-gigaword-50'))

# Finds most similar items to given example:
def find_similar(word: str):
  w2v_model = downloader.load("glove-wiki-gigaword-50")
  return w2v_model.most_similar(word.lower())

def find_similar_to_jupiter():
  return find_similar("Jupiter")
  
def test_find_similar():
  print("blue...")
  answers = find_similar("blue")
  assert(answers[0][0]) == "red"
  print("king...")
  answers = find_similar("king")
  assert(answers[0][0]) == "prince"
  print("sweden...")
  answers = find_similar("sweden")
  assert(answers[0][0]) == "denmark"
  print("estonia...")
  answers = find_similar("estonia")
  assert(answers[0][0]) == "latvia"