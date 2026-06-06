from pathlib import Path
from typing import Any
from pydantic import BaseModel, model_validator
from langchain_text_splitters import RecursiveCharacterTextSplitter, RecursiveJsonSplitter
from langchain_text_splitters import MarkdownTextSplitter, PythonCodeTextSplitter
from .tools import MinimalSource


class Chunker(BaseModel):
    data_base: Path
    data_list: list[Path] = []
    data_list_md: list[Path] = []
    data_list_py: list[Path] = []
    data_list_json: list[Path] = []

    @model_validator(mode='after')
    def parsing_database(self) -> 'Chunker':
        for elem in self.data_base.rglob('*'):
            if elem.suffix == '.md':
                self.data_list_md.append(elem)
            elif elem.suffix == '.py':
                self.data_list_py.append(elem)
            elif elem.suffix == '.json':
                self.data_list_json.append(elem)
            else:
                self.data_list.append(elem)
        for elem in self.data_list:
            print(elem)
        return self

    def chunking(self) -> list[MinimalSource]:
        list_chunkers: list[MinimalSource] = []
        try:
            for elem in self.data_list_md:
                



if __name__ == '__main__':
    ch = Chunker(data_base=Path('/home/gematura/cursus_42/tronc_commum/rag/data/vllm-0.10.1/'))
