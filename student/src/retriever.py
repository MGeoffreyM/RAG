import json
import uuid
import bm25s
from tqdm import tqdm
import Stemmer
import chromadb
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from src.tools import (MinimalSource,
                       MinimalSearchResults,
                       StudentSearchResults,
                       MinimalAnswer, StudentSearchResultsAndAnswer)


class Retriever(BaseModel):
    index_dir: Path
    chunks_file: Path
    _llm_instance: Any = None
    _llm_embedding: Any = None

    def _call_qwen_embedding(self) -> Any:
        if self._llm_embedding is None:
            self._llm_embedding = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2')
        return self._llm_embedding

    def save_index(self, src: list[MinimalSource]) -> None:
        """Crée l'index BM25 avec l'Ancre Sémantique Totale."""
        prepared_data: list[dict[str, Any]] = []
        for i, elem in enumerate(src):
            path_keywords = elem.file_path.replace(
                "/", " ").replace("_", " ").replace(".", " ").replace("-", " ")

            prepared_data.append({
                "id": str(i),
                "bm25_content": f"{path_keywords}\n{elem.chunk_content}",
                "document": elem.chunk_content,
                "metadata": {"file_path": elem.file_path},
                "source_dump": elem.model_dump()
            })

        ids = [item["id"] for item in prepared_data]
        bm25_contents = [item["bm25_content"] for item in prepared_data]
        documents = [item["document"] for item in prepared_data]
        metadatas = [item["metadata"] for item in prepared_data]

        # embedding
        model = self._call_qwen_embedding()
        corpus_embeddings = model.encode(bm25_contents)

        # bm25
        stemmer = Stemmer.Stemmer("english")
        corpus_bm25 = bm25s.tokenize(bm25_contents, stopwords="en",
                                     stemmer=stemmer)

        retriever = bm25s.BM25(k1=1.5, b=0.75)
        retriever.index(corpus_bm25)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(self.index_dir))

        chroma_client = chromadb.PersistentClient(
            path=str(self.index_dir / "chroma_db"))
        collection = chroma_client.get_or_create_collection(
            name="mes_documents")

        BATCH_SIZE = 5000

        for i in tqdm(range(0, len(ids), BATCH_SIZE),
                      desc="recherche semantique"):
            end_idx = i + BATCH_SIZE
            collection.add(
                ids=ids[i:end_idx],
                embeddings=corpus_embeddings[i:end_idx].tolist(),
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

        self.chunks_file.parent.mkdir(parents=True, exist_ok=True)
        chunks_data = [chunk.model_dump() for chunk in src]
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=4)

    def search_query(self, question: str, k: int = 1,
                     question_id: str | None = None) -> StudentSearchResults:
        if question_id is None:
            question_id = str(uuid.uuid4())

        model = self._call_qwen_embedding()
        query_embedding = model.encode(question)

        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize(question, stopwords="en",
                                      stemmer=stemmer)

        retriever = bm25s.BM25.load(str(self.index_dir), load_corpus=False)

        chroma_client = chromadb.PersistentClient(
            path=str(self.index_dir / "chroma_db"))
        collection = chroma_client.get_collection(name="mes_documents")

        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        search_limit = max(k * 10, 1000)
        bm25_results, bm25_scores = retriever.retrieve(query_tokens,
                                                       k=search_limit)

        chroma_results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=search_limit
        )

        rrf_scores: dict[str, float] = {}
        K = 10

        for rank, doc_index in enumerate(bm25_results[0]):
            # On convertit simplement l'index BM25 en string
            doc_id = str(doc_index)
            rrf_scores[doc_id] = 0.60 * (1 / (K + rank + 1))

        for rank, doc_id in enumerate(chroma_results['ids'][0]):
            if doc_id in rrf_scores:
                rrf_scores[doc_id] += 0.40 * (1 / (K + rank + 1))
            else:
                rrf_scores[doc_id] = 0.40 * (1 / (K + rank + 1))

        # 6. Tri des résultats par le meilleur score RRF
        sorted_ids = sorted(rrf_scores.keys(),
                            key=lambda x: rrf_scores[x],
                            reverse=True)
        top_ids = sorted_ids[:k]

        retrieved_sources: list[MinimalSource] = []

        for str_idx in top_ids:
            chunk_dict = chunks_data[int(str_idx)]
            source = MinimalSource(**chunk_dict)
            retrieved_sources.append(source)

        minimal_result = MinimalSearchResults(
            question_id=question_id,
            question=question,
            retrieved_sources=retrieved_sources
        )

        return StudentSearchResults(search_results=[minimal_result], k=k)

    def generate_answer(self, question: str,
                        sources: list[MinimalSource]) -> Any:
        """Génère une réponse textuelle à partir de sources déjà trouvées."""
        context = ""
        for i, source in enumerate(sources):
            context += f"--- Source {i+1} ({source.file_path}) ---\n"
            context += f"{source.chunk_content}\n\n"

        prompt = f"""You are an expert technical development assistant.
                     Your task is to answer the question based SOLELY on the
                     provided context.
                     If the context does not contain the information needed to
                     answer the question, you must answer exactly: "I don't
                     know based on the provided context." Don't make any
                     assumptions.
                     Context: {context}, Question: {question}, Answer:"""
        response, thinking = self._call_qwen_model(prompt)
        print(thinking)
        return response

    def _call_qwen_model(self, prompt: str) -> Any:
        if self._llm_instance is None:
            # Lazy import
            from src.llm import Llm
            self._llm_instance = Llm("Qwen/Qwen3-0.6B")

        return self._llm_instance.prompting(prompt)

    def answer_query(self, question: str, k: int = 1,
                     question_id: str | None = None
                     ) -> StudentSearchResultsAndAnswer:
        """Pour la commande 'answer' (1 seule question) : Cherche ET répond."""
        # 1. On cherche
        search_res = self.search_query(question, k, question_id)
        minimal_search = search_res.search_results[0]

        # 2. On génère la réponse
        answer_text = self.generate_answer(minimal_search.question,
                                           minimal_search.retrieved_sources)

        # 3. On formate
        minimal_answer = MinimalAnswer(
            question_id=minimal_search.question_id,
            question=minimal_search.question,
            retrieved_sources=minimal_search.retrieved_sources,
            answer=answer_text
        )
        return StudentSearchResultsAndAnswer(search_results=[minimal_answer],
                                             k=k)
