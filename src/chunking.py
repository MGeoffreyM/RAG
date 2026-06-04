from pathlib import Path
from pydantic import BaseModel, Field
from .tools import MinimalSource


def code_chunking(content:str) -> MinimalSource:
    while len(content > 0):



def chunking(file:str) -> MinimalSource | None:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            pat = Path(file)
            content = f.read()
            if len(content) == 0:
                return
            elif 0 < len(content) <= 2000:
                return MinimalSource(file_path=Path(file).resolve(),
                                     first_character_index=0,
                                     last_character_index=len(content) - 1)
            elif len(content) > 2000 and pat.suffix == '.py':
                