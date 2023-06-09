# (C) Heikki Kupiainen 2023    

from typing import List
from pydantic import BaseModel, StrictStr

class StringList(BaseModel):
  items: List[StrictStr]
  
  @staticmethod
  def new(strings: list[str]):
    return StringList(**{
      'items': strings
    })

def test_instantiation():
  assert(type(StringList.new(['list', 'of', 'words']))) == StringList