# (C) Heikki Kupiainen 2023    

from typing import List
from pydantic import BaseModel, StrictStr

class StringList(BaseModel):
  items: List[StrictStr]

def test_instantiation():
  assert(type((StringL(**{
    'items': ['list', 'of', 'words']
  })))) == StringList