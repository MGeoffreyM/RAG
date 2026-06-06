from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


dossier = Path('/home/gematura/cursus_42/tronc_commum/rag/'
               'data/vllm-0.10.1')


if __name__ == '__main__':
    test = [elem for elem in dossier.rglob('*.py')]
    for eleme in test:
        print(eleme)
