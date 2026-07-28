import json
import os
from pathlib import Path
from tqdm import tqdm
from typing import Annotated
from pydantic import Field, validate_call, StringConstraints
from src.evaluator import Evaluator
from src.chunking import Chunker
from src.retriever import Retriever
from src.tools import (UnansweredQuestion,
                       RagDataset,
                       MinimalSearchResults,
                       MinimalAnswer,
                       StudentSearchResults,
                       StudentSearchResultsAndAnswer)


class RagCLI:
    """ Interface en ligne de commande"""
    def __init__(self, semantic: bool = False) -> None:
        self.index_path = Path("data/processed/bm25_index")
        self.chunks_path = Path("data/processed/chunks/chunks.json")
        self.rt = Retriever(index_dir=self.index_path,
                            chunks_file=self.chunks_path,
                            semantic=semantic)

    def index(self,
              data_path: Path = Path(
                  'data/raw/vllm-0.10.1/'),
              max_chunk_size: int = 2000,
              image: bool = False,
              force: bool = False) -> None:
        """
        Indexe la base de données et crée les chunks.

        Args:
            data_path: chemin de la base de donnée (default: vllm-0.10.1),
            max_chunk_size: La taille maximale d'un chunk (défaut: 2000),
            image: permet le décriptage des images (défault: False),
            force: permet de forcer la réindexation (défault: False).
        """
        state_file = Path("data/processed/index_state.json")
        old_state = {}
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                old_state = json.load(f)

        new_state = {}
        new_files = []

        try:
            for filepath in data_path.rglob('*'):
                if filepath.is_file():
                    mtime = os.path.getmtime(filepath)
                    str_path = str(filepath).replace("\\", "/")
                    new_state[str_path] = mtime

                    if str_path not in old_state or old_state[
                            str_path] < mtime:
                        new_files.append(filepath)

            deleted_files = [f for f in old_state if f not in new_state]

            files_to_remove = (
                [str(f).replace("\\", "/") for f in new_files] + deleted_files)

            chroma_db_dir = self.rt.index_dir / "chroma_db"
            needs_semantic_catchup = self.rt.semantic and not chroma_db_dir.exists()

            # Si on a pas de fichiers à enlever ET qu'on ne force pas ET qu'on n'a pas besoin de rattrapage
            if not files_to_remove and not force and not needs_semantic_catchup:
                print("Aucun fichier n'a été modifié. "
                      "L'index est déjà à jour.")
                return

            print(f"Indexation : {len(new_files)} fichiers modifiés/ajoutés,"
                  f" {len(deleted_files)} supprimés.")

            ch = Chunker(data_base=data_path, new_files=new_files)
            ch.process_text()
            if image:
                ch.process_images()
            ch.create_smart_chunks(chunk_size=max_chunk_size)

            if not old_state:
                self.rt.save_index(ch.list_chunk)
            else:
                self.rt.update_index(ch.list_chunk, files_to_remove)

            state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(new_state, f, indent=4)

        except Exception as e:
            print(f"Erreur à l'indexation : {e}")

    @validate_call
    def search(self,
               query: Annotated[str, StringConstraints(min_length=1,
                                                       strip_whitespace=True)],
               k: Annotated[int, Field(gt=0)] = 10) -> None:
        try:
            question = UnansweredQuestion(question=query)
            result = self.rt.search_query(question, k)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Erreur à la recherche : {e}")

    @validate_call
    def search_dataset(self,
                       dataset_path: str | Path,
                       save_directory: str | Path = (
                           'data/output/search_results/'),
                       k: Annotated[int, Field(gt=0)] = 10) -> None:

        dataset_path = Path(dataset_path)
        save_directory = Path(save_directory)

        try:
            list_questions = RagDataset(rag_questions=[])
            with open(dataset_path, 'r', encoding='utf-8') as f:
                e = json.load(f)
            for elem in e.get("rag_questions", []):
                question_data = {"question": elem["question"]}
                q_id = elem.get("question_id")
                question_data["question_id"] = q_id

                list_questions.rag_questions.append(
                    UnansweredQuestion(**question_data))

            minimal_results: list[MinimalSearchResults] = []
            for quest in tqdm(list_questions.rag_questions,
                              desc="recherche des sources"):
                src = self.rt.search_query(quest, k)
                minimal_results.append(src.search_results[0])

            final_results = StudentSearchResults(
                search_results=minimal_results,
                k=k
            )

            save_directory.mkdir(parents=True, exist_ok=True)
            file_path = save_directory / dataset_path.name
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(final_results.model_dump_json(indent=2))
        except Exception as e:
            print(f"Erreur à la recherche groupée : {e}")

    @validate_call
    def answer(self,
               query: Annotated[str, StringConstraints(min_length=1,
                                                       strip_whitespace=True)],
               k: Annotated[int, Field(gt=0)] = 10) -> None:
        try:
            question = UnansweredQuestion(question=query)
            result = self.rt.answer_query(question, k)
            print(result.model_dump_json(indent=2))
            with open("data/processed/answer.json", "w",
                      encoding="utf-8") as r:
                r.write(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Erreur à la question : {e}")

    @validate_call
    def answer_dataset(self,
                       student_search_results_path: str | Path,
                       save_directory: str | Path = (
                           'data/output/search_results_and_answer/'
                       ),
                       k: Annotated[int, Field(gt=0)] = 10) -> None:

        student_search_results_path = Path(student_search_results_path)
        save_directory = Path(save_directory)

        try:
            with open(student_search_results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_k = data.get("k", k)

            minimal_answers: list[MinimalAnswer] = []

            for item in tqdm(data.get("search_results", []),
                             desc="Génération des réponses"):
                search_res = MinimalSearchResults(**item)

                answer_text = self.rt.generate_answer(
                    search_res.question, search_res.retrieved_sources)

                ans = MinimalAnswer(
                    question_id=search_res.question_id,
                    question=search_res.question,
                    retrieved_sources=search_res.retrieved_sources,
                    answer=answer_text
                )
                minimal_answers.append(ans)

            final_results = StudentSearchResultsAndAnswer(
                search_results=minimal_answers,
                k=file_k
            )

            save_directory.mkdir(parents=True, exist_ok=True)
            file_path = save_directory / student_search_results_path.name
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(final_results.model_dump_json(indent=2))

            print(f"✅ Réponses sauvegardées avec succès dans {file_path}")

        except Exception as e:
            print(f'❌ Erreur à la génération des réponses : {e}')

    @validate_call
    def evaluate(self,
                 student_search_results_path: str | Path,
                 dataset_path: str | Path,
                 k: Annotated[int, Field(gt=0)] = 10) -> None:
        """Évalue les résultats de recherche contre le Ground Truth."""

        ev = Evaluator(
            student_search_results_path=Path(student_search_results_path),
            dataset_path=Path(dataset_path)
        )
        ev.run_evaluation(k=k)

    def serve(self, port: int = 8000) -> None:
        """Lance l'API HTTP locale."""
        import uvicorn
        print(f"Lancement du serveur sur http://localhost:{port}")
        uvicorn.run("src.api:app", host="127.0.0.1", port=port, reload=False)
