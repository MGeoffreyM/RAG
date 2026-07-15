import json
import uuid
import bm25s
import dspy
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from src.llm import AnswerBot
from src.tools import (MinimalSource,
                       MinimalSearchResults,
                       StudentSearchResults,
                       MinimalAnswer, StudentSearchResultsAndAnswer)


class Retriever(BaseModel):
    index_dir: Path
    chunks_file: Path
    _llm_instance: Any = None

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

    def search_query(self, question: str, k: int = 1,
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
            question_str=question,
            retrieved_sources=retrieved_sources
        )

        return StudentSearchResults(
            search_results=[minimal_result],
            k=k
        )

    def generate_answer(self, question: str,
                        sources: list[MinimalSource]) -> Any:
        """Génère une réponse textuelle à partir de sources déjà trouvées."""
        context = ""
        for i, source in enumerate(sources):
            context += f"--- Source {i+1} ({source.file_path}) ---\n"
            context += f"{source.chunk_content}\n\n"
        self._call_qwen_model()
        answer_bot = dspy.Predict(AnswerBot)
        answer = answer_bot(question=question, context=context)

        return answer.answer

    def _call_qwen_model(self) -> None:
        name: str = "openai/Qwen/Qwen3-0.6B"
        api_key: str = "sk-fake-key"
        if self._llm_instance is None:
            self._llm_instance = dspy.LM(name,
                                         api_base="http://localhost:8000/v1",
                                         api_key=api_key)
            dspy.configure(lm=self._llm_instance)

    def answer_query(self, question: str, k: int = 1,
                     question_id: str | None = None
                     ) -> StudentSearchResultsAndAnswer:
        """Pour la commande 'answer' (1 seule question) : Cherche ET répond."""
        # 1. On cherche
        search_res = self.search_query(question, k, question_id)
        minimal_search = search_res.search_results[0]

        # 2. On génère la réponse
        answer_text = self.generate_answer(minimal_search.question_str,
                                           minimal_search.retrieved_sources)

        # 3. On formate
        minimal_answer = MinimalAnswer(
            question_id=minimal_search.question_id,
            question_str=minimal_search.question_str,
            retrieved_sources=minimal_search.retrieved_sources,
            answer=answer_text
        )
        return StudentSearchResultsAndAnswer(search_results=[minimal_answer],
                                             k=k)
