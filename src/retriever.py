import bm25s
from chunking import Chunker
from tools import MinimalSource


class Retriever():

    def save_index(self, src: list[MinimalSource]) -> None:
        corpus: list[str] = []
        for elem in src:
            corpus.append(elem.chunk_content)

        corpus_tokens = bm25s.tokenize(corpus)
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        retriever.save("data/processed/bm25_index")
