import json
import uuid
import bm25s
import Stemmer
import chromadb
from pathlib import Path
from typing import Any
from tqdm import tqdm
from pydantic import BaseModel
from chromadb.api.types import Metadata
from src.tools import (MinimalSource,
                       UnansweredQuestion,
                       MinimalSearchResults,
                       MinimalAnswer,
                       StudentSearchResults,
                       StudentSearchResultsAndAnswer)


class Retriever(BaseModel):
    index_dir: Path
    chunks_file: Path
    cache_file: Path = Path("data/processed/cache.json")
    namespace: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "rag.42.fr")
    _llm_instance: Any = None
    _llm_embedding: Any = None
    batch_size: int = 5000
    semantic: bool

    def _get_cache(self) -> dict[str, Any] | Any:
        """Charge le cache depuis le fichier."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"search": {}, "answer": {}}

    def _save_cache(self, cache_data: dict[str, Any]) -> None:
        """Sauvegarde le cache sur le fichier."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)

    def _generate_cache_key(self, question: str, k: int) -> str:
        """Crée un UUID unique et déterministe basé sur la question et k."""
        payload = f"{question}_{k}"
        cache_uuid = uuid.uuid5(self.namespace, payload)
        return str(cache_uuid)

    def _call_model_embedding(self) -> Any:
        if self._llm_embedding is None:
            from sentence_transformers import SentenceTransformer
            self._llm_embedding = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2')
        return self._llm_embedding

    def _call_qwen_model(self, prompt: str) -> Any:
        if self._llm_instance is None:
            from src.llm import AnswerLlm
            self._llm_instance = AnswerLlm("Qwen/Qwen3-0.6B")
        return self._llm_instance.prompting(prompt)

    def save_index(self,
                   src: list[MinimalSource]) -> None:
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
        metadatas: list[Metadata] = [
            item["metadata"] for item in prepared_data]

        # bm25
        stemmer = Stemmer.Stemmer("english")
        corpus_bm25 = bm25s.tokenize(bm25_contents,
                                     stopwords="en",
                                     stemmer=stemmer)
        retriever = bm25s.BM25(k1=1.5, b=0.75)
        retriever.index(corpus_bm25)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(self.index_dir))

        # embedding
        if self.semantic:
            model = self._call_model_embedding()
            corpus_embeddings = model.encode(bm25_contents)

            chroma_client = chromadb.PersistentClient(
                path=str(self.index_dir / "chroma_db"))

            try:
                chroma_client.delete_collection(name="mes_documents")
            except ValueError:
                pass

            collection = chroma_client.create_collection(name="mes_documents")

            for i in tqdm(range(0, len(ids), self.batch_size),
                          desc="Recherche sémantique"):
                end_idx = i + self.batch_size
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

    def update_index(self,
                     new_chunks: list[MinimalSource],
                     files_to_remove: list[str]) -> None:
        """Met à jour l'index des fichiers modifiés (ChromaDB + BM25)."""

        all_chunks_dict = []
        if self.chunks_file.exists():
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                all_chunks_dict = json.load(f)

        remaining_chunks = [c for c in all_chunks_dict
                            if c["file_path"] not in files_to_remove]
        new_chunks_dict = [c.model_dump() for c in new_chunks]
        updated_chunks_dict = remaining_chunks + new_chunks_dict

        if self.semantic:
            chroma_client = chromadb.PersistentClient(
                path=str(self.index_dir / "chroma_db"))
            collection = chroma_client.get_or_create_collection(
                name="mes_documents")

            if collection.count() == 0 and remaining_chunks:
                print("Index sémantique manquant. Rattrapage...")
                model = self._call_model_embedding()

                old_bm25 = [f"{c['file_path']}\n{c['chunk_content']}"
                            for c in remaining_chunks]
                old_docs = [c['chunk_content'] for c in remaining_chunks]
                old_metadatas = [{"file_path": c['file_path']}
                                 for c in remaining_chunks]
                old_ids = [str(uuid.uuid4()) for _ in remaining_chunks]

                old_embeddings = model.encode(old_bm25).tolist()

                for i in tqdm(range(0, len(old_ids), self.batch_size),
                              desc="Rattrapage ChromaDB"):
                    end_idx = i + self.batch_size
                    collection.add(
                        ids=old_ids[i:end_idx],
                        embeddings=old_embeddings[i:end_idx],
                        documents=old_docs[i:end_idx],
                        metadatas=old_metadatas[i:end_idx]
                    )

            for file_path in files_to_remove:
                collection.delete(where={"file_path": file_path})

            if new_chunks:
                if 'model' not in locals():
                    model = self._call_model_embedding()

                new_bm25_contents = [f"{c.file_path}\n{c.chunk_content}"
                                     for c in new_chunks]
                print("Calcul des embeddings pour les nouveaux fichiers...")
                new_embeddings = model.encode(new_bm25_contents).tolist()
                documents = [c.chunk_content for c in new_chunks]
                metadatas = [{"file_path": c.file_path} for c in new_chunks]

                new_ids = [str(uuid.uuid4()) for _ in new_chunks]

                for i in tqdm(range(0, len(new_ids), self.batch_size),
                              desc="Mise à jour ChromaDB"):
                    end_idx = i + self.batch_size
                    collection.add(
                        ids=new_ids[i:end_idx],
                        embeddings=new_embeddings[i:end_idx],
                        documents=documents[i:end_idx],
                        metadatas=metadatas[i:end_idx]
                    )

        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            json.dump(updated_chunks_dict, f, indent=4)

        if not updated_chunks_dict:
            print("Aucun document restant dans l'index.")
            return

        bm25_contents = [f"{c['file_path']}\n{c['chunk_content']}"
                         for c in updated_chunks_dict]
        stemmer = Stemmer.Stemmer("english")
        corpus_bm25 = bm25s.tokenize(bm25_contents,
                                     stopwords="en",
                                     stemmer=stemmer)
        retriever = bm25s.BM25(k1=1.5, b=0.75)
        retriever.index(corpus_bm25)
        retriever.save(str(self.index_dir))

    def search_query(self, query: UnansweredQuestion,
                     k: int) -> StudentSearchResults:
        cache_data = self._get_cache()
        cache_key = self._generate_cache_key(query.question, k)
        if cache_key in cache_data["search"]:
            return StudentSearchResults(**cache_data["search"][cache_key])

        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize(query.question,
                                      stopwords="en",
                                      stemmer=stemmer)
        retriever = bm25s.BM25.load(str(self.index_dir), load_corpus=False)

        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)

        search_limit = max(k * 10, 1000)
        bm25_results, _ = retriever.retrieve(query_tokens, k=search_limit)

        if self.semantic:
            chroma_client = chromadb.PersistentClient(
                path=str(self.index_dir / "chroma_db"))
            collection = chroma_client.get_or_create_collection(
                name="mes_documents")

            if collection.count() == 0 and chunks_data:
                model = self._call_model_embedding()

                bm25_contents = [
                    f"{c['file_path']}\n{c['chunk_content']}"
                    for c in chunks_data]
                documents = [c['chunk_content'] for c in chunks_data]
                metadatas = [
                    {"file_path": c['file_path']} for c in chunks_data]
                ids = [str(i) for i in range(len(chunks_data))]

                corpus_embeddings = model.encode(bm25_contents)

                for i in tqdm(range(0, len(ids), self.batch_size),
                              desc="ChromaDB"):
                    end_idx = i + self.batch_size
                    collection.add(
                        ids=ids[i:end_idx],
                        embeddings=corpus_embeddings[i:end_idx].tolist(),
                        documents=documents[i:end_idx],
                        metadatas=metadatas[i:end_idx]
                    )
            else:
                model = self._call_model_embedding()

            query_embedding = model.encode(query.question)
            chroma_results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=search_limit
            )

            rrf_scores: dict[str, float] = {}
            K = 10

            for rank, doc_index in enumerate(bm25_results[0]):
                doc_id = str(doc_index)
                rrf_scores[doc_id] = 0.60 * (1 / (K + rank + 1))

            for rank, doc_id in enumerate(chroma_results['ids'][0]):
                if doc_id in rrf_scores:
                    rrf_scores[doc_id] += 0.40 * (1 / (K + rank + 1))
                else:
                    rrf_scores[doc_id] = 0.40 * (1 / (K + rank + 1))

            sorted_ids = sorted(rrf_scores.keys(),
                                key=lambda x: rrf_scores[x],
                                reverse=True)
            top_ids = sorted_ids[:k]

        else:
            top_ids = [str(idx) for idx in bm25_results[0][:k]]

        retrieved_sources: list[MinimalSource] = []

        for str_idx in top_ids:
            chunk_dict = chunks_data[int(str_idx)]
            source = MinimalSource(**chunk_dict)
            retrieved_sources.append(source)

        minimal_result = MinimalSearchResults(
            question_id=query.question_id,
            question=query.question,
            retrieved_sources=retrieved_sources
        )

        final_result = StudentSearchResults(search_results=[minimal_result],
                                            k=k)

        cache_data["search"][cache_key] = final_result.model_dump()
        self._save_cache(cache_data)

        return final_result

    def generate_answer(self, question: str,
                        sources: list[MinimalSource]) -> Any:
        """Génère une réponse textuelle à partir de sources déjà trouvées."""
        context = ""
        for i, source in enumerate(sources):
            context += f"--- Source {i+1} ({source.file_path}) ---\n"
            context += f"{source.chunk_content}\n\n"
        prompt = f"""Answer the question based ONLY on the provided context.
If the context does not contain the answer, reply exactly: "I don't know."

--- CONTEXT ---
{context.strip()}

--- QUESTION ---
{question}

--- ANSWER ---
"""
        answer = self._call_qwen_model(prompt)
        return answer[0]

    def answer_query(self, question: UnansweredQuestion,
                     k: int) -> StudentSearchResultsAndAnswer:
        """Pour la commande 'answer' (1 seule question) : Cherche ET répond."""
        cache_data = self._get_cache()
        cache_key = self._generate_cache_key(question.question, k)

        if cache_key in cache_data["answer"]:
            return StudentSearchResultsAndAnswer(
                **cache_data["answer"][cache_key])

        search_res = self.search_query(question, k)
        minimal_search = search_res.search_results[0]

        answer_text = self.generate_answer(minimal_search.question,
                                           minimal_search.retrieved_sources)

        minimal_answer = MinimalAnswer(
            question_id=minimal_search.question_id,
            question=minimal_search.question,
            retrieved_sources=minimal_search.retrieved_sources,
            answer=answer_text
        )
        final_result = StudentSearchResultsAndAnswer(
            search_results=[minimal_answer], k=k)

        cache_data["answer"][cache_key] = final_result.model_dump()
        self._save_cache(cache_data)

        return final_result
