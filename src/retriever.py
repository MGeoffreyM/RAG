import json
import uuid
import bm25s
from pathlib import Path
from pydantic import BaseModel
from src.llm import Llm
from src.tools import (MinimalSource,
                       MinimalSearchResults,
                       StudentSearchResults,
                       MinimalAnswer, StudentSearchResultsAndAnswer)


class Retriever(BaseModel):
    index_dir: Path
    chunks_file: Path

    def save_index(self, src: list[MinimalSource]) -> None:
        """Crée l'index BM25 et sauvegarde les métadonnées (chunks)."""
        corpus: list[str] = [elem.chunk_content for elem in src]

        corpus_tokens = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(self.index_dir))

        self.chunks_file.parent.mkdir(parents=True, exist_ok=True)

        chunks_data = [chunk.model_dump() for chunk in src]

        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=4)

    def search_query(self, question: str, k: int = 10,
                     question_id: str | None = None) -> StudentSearchResults:
        """
        Charge l'index, recherche les meilleurs chunks pour une question
        et retourne le JSON formaté selon les normes du sujet.
        """
        if question_id is None:
            question_id = str(uuid.uuid4())

        retriever = bm25s.BM25.load(str(self.index_dir), load_corpus=False)

        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        query_tokens = bm25s.tokenize(question)

        results, scores = retriever.retrieve(query_tokens, k=k)

        retrieved_sources: list[MinimalSource] = []

        for idx in results[0]:
            chunk_dict = chunks_data[idx]
            source = MinimalSource(**chunk_dict)
            retrieved_sources.append(source)

        minimal_result = MinimalSearchResults(
            question_id=question_id,
            question=question,
            retrieved_sources=retrieved_sources
        )

        return StudentSearchResults(
            search_results=[minimal_result],
            k=k
        )

    def answer_query(self, question: str,
                     k: int = 10,
                     question_id: str = None) -> StudentSearchResultsAndAnswer:
        llm = Llm("Qwen/Qwen3-0.6B")
        search_result = self.search_query(question, k, question_id)
        minimal_search = search_result.search_results[0]

        context = ""
        for i, source in enumerate(minimal_search.retrieved_sources):
            context += f"--- Source {i+1} ({source.file_path}) ---\n"
            context += f"{source.chunk_content}\n\n"

        prompt = f"""You are an expert technical development assistant.
        Your task is to answer the question based SOLELY on the provided
        context.
        If the context does not contain the information needed to answer the
        question, you must answer exactly: "I don't know based on the provided
        context."
        Don't make any assumptions.
        Context:
        {context}
        Question: {question}
        Answer:"""

        response = llm.prompting(prompt)

        minimal_answer = MinimalAnswer(
            question_id=minimal_search.question_id,
            question=minimal_search.question,
            retrieved_sources=minimal_search.retrieved_sources,
            answer=response
        )

        return StudentSearchResultsAndAnswer(
            search_results=[minimal_answer],
            k=k
        )
