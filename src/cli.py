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
        """
        Initialise les chemins d'index et de chunks, ainsi que le Retriever.

        Args:
            semantic (bool): Active la recherche sémantique (défaut: False).
        """
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

        Cette méthode vérifie l'état précédent de l'index et ne traite que les
        fichiers nouveaux ou modifiés, sauf si l'argument `force` est utilisé.

        Args:
            data_path (Path): Chemin du répertoire de la base de données
                              (défaut: 'data/raw/vllm-0.10.1/').
            max_chunk_size (int): La taille maximale autorisée pour un chunk
                                  (défaut: 2000).
            image (bool): Permet le décryptage et l'analyse des images si True
                          (défaut: False).
            force (bool): Force la réindexation complète de tous les fichiers
                          (défaut: False).
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
            semantic_catchup = self.rt.semantic and not chroma_db_dir.exists()

            if not files_to_remove and not force and not semantic_catchup:
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
        """
        Effectue une recherche d'information à partir d'une requête et affiche
        les résultats au format JSON.

        Args:
            query (str): La requête de recherche, qui doit comporter au moins
                         1 caractère et être sans espaces superflus.
            k (int): Le nombre maximum de résultats à retourner,
                     doit être strictement supérieur à 0 (défaut: 10).
        """
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
        """
        Parcourt un dataset de questions et effectue une recherche pour
        chacune d'elles.

        Les résultats obtenus pour l'ensemble des questions sont ensuite
        sauvegardés dans un fichier JSON.

        Args:
            dataset_path (str | Path): Le chemin vers le fichier JSON
                                       contenant le dataset de questions.
            save_directory (str | Path): Le répertoire de destination pour
                                    sauvegarder les résultats
                                    (défaut: 'data/output/search_results/').
            k (int): Le nombre de sources à récupérer pour chaque question,
                     doit être strictement supérieur à 0 (défaut: 10).
        """
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
        """
        Recherche des sources pertinentes pour une question,
        puis génère et sauvegarde une réponse.

        La réponse générée est affichée dans la console et sauvegardée
        localement dans 'data/processed/answer.json'.

        Args:
            query (str): La question posée, qui doit comporter au moins
                         1 caractère (sans espaces superflus).
            k (int): Le nombre de sources à utiliser pour formuler la réponse
                     (défaut: 10).
        """
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
        """
        Génère des réponses en lot pour un ensemble de résultats de recherche
        existants issus d'un dataset.

        Chaque réponse est générée à partir des sources récupérées,
        puis l'ensemble est sauvegardé au format JSON.

        Args:
            student_search_results_path (str | Path): Le chemin vers le
                fichier contenant les résultats de recherche.
            save_directory (str | Path): Le dossier où sauvegarder le fichier
                final contenant les réponses
                (défaut: 'data/output/search_results_and_answer/').
            k (int): Le nombre de sources considérées, doit être strictement
                supérieur à 0 (défaut: 10).
        """
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
        """
        Évalue les résultats de recherche de l'étudiant en les comparant avec
        le Ground Truth d'un dataset de référence.

        Args:
            student_search_results_path (str | Path): Le chemin vers le
                fichier contenant les résultats de recherche à évaluer.
            dataset_path (str | Path): Le chemin vers le fichier du dataset
                contenant les questions et réponses de référence.
            k (int): Le nombre maximum de résultats pris en compte pour
                calculer les métriques d'évaluation (défaut: 10).
        """

        ev = Evaluator(
            student_search_results_path=Path(student_search_results_path),
            dataset_path=Path(dataset_path)
        )
        ev.run_evaluation(k=k)

    def serve(self, port: int = 8000) -> None:
        """
        Lance l'API HTTP locale en utilisant Uvicorn.

        Args:
            port (int): Le numéro de port sur lequel exécuter le serveur
                        (défaut: 8000).
        """
        import uvicorn
        print(f"Lancement du serveur sur http://localhost:{port}")
        uvicorn.run("src.api:app", host="127.0.0.1", port=port, reload=False)
