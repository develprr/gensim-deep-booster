# (C) Heikki Kupiainen 2023    

from typing import List
from pydantic import BaseModel, StrictStr

class StringList(BaseModel):
  items: List[StrictStr]
  
  @staticmethod
  def build(strings: list[str]):
    return StringList(**{
      'items': ['list', 'of', 'words']
    })


def test_instantiation():
  assert(type(StringList.build(['list', 'of', 'words']))) == StringList