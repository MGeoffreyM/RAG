import bm25s
from chunking import Chunker
from pathlib import Path


# Create your corpus here
corpus = []
for elem in ch.list_chunk:
    corpus.append(elem.chunk_content)

# Tokenize the corpus and index it
corpus_tokens = bm25s.tokenize(corpus)
retriever = bm25s.BM25(corpus=corpus)
retriever.index(corpus_tokens)

# You can now search the corpus with a query
query = "python and llm?"
query_tokens = bm25s.tokenize(query)
docs, scores = retriever.retrieve(query_tokens, k=2)
print(f"Best result (score: {scores[0, 0]:.2f}): {docs[0, 0]}")

# Happy with your index? Save it for later...
retriever.save("bm25s_index_animals")

# ...and load it when needed
ret_loaded = bm25s.BM25.load("bm25s_index_animals", load_corpus=True)