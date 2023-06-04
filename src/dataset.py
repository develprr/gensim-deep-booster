# (C) Heikki Kupiainen 2023    

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, StrictStr, StrictInt

class StringDataset(BaseModel):
  items: List[List[StrictStr]]

def test_instantiation():
  assert(type((StringDataset(**{
    'items': [
      ['list', 'of', 'words']
    ]
  })))) == StringDataset