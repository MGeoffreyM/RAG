import json
import fire
from pathlib import Path
from tqdm import tqdm
from src.chunking import Chunker
from src.retriever import Retriever
from src.tools import (UnansweredQuestion,
                       StudentSearchResults,
                       MinimalSearchResults,
                       MinimalAnswer,
                       StudentSearchResultsAndAnswer)


class RagCLI:
    """ Interface en ligne de commande"""
    def __init__(self) -> None:
        self.index_path = Path("data/processed/bm25_index")
        self.chunks_path = Path("data/processed/chunks/chunks.json")
        self.rt = Retriever(
            index_dir=self.index_path, chunks_file=self.chunks_path)

    def index(self,
              data_path: Path = Path(
                  'data/raw/vllm-0.10.1/'),
              max_chunk_size: int = 800,
              index_image: bool = False) -> None:
        """
        Indexe la base de données et crée les chunks.

        Args:
            data_path: chemin de la base de donnée (default: vllm-0.10.1),
            max_chunk_size: La taille maximale d'un chunk (défaut: 2000),
            index_image: permet le décriptage des images (défault: False).
        """
        try:
            ch = Chunker(data_base=data_path)
            ch.process_text()
            if index_image:
                ch.process_images()
            ch.create_smart_chunks(chunk_size=max_chunk_size)
            self.rt.save_index(ch.list_chunk)

        except Exception as e:
            print(f"Erreur à l'indexation : {e}")

    def search(self,
               question: str,
               k: int = 10,
               question_id: str | None = None) -> None:
        try:
            result = self.rt.search_query(question, k, question_id)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Erreur à la recherche : {e}")

    def search_dataset(self,
                       dataset_path: str | Path,
                       save_directory: str | Path = (
                           'data/output/search_results/'),
                       k: int = 10) -> None:

        dataset_path = Path(dataset_path)
        save_directory = Path(save_directory)

        try:
            list_questions: list[UnansweredQuestion] = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                e = json.load(f)
            for elem in e.get("rag_questions", []):
                question_data = {"question": elem["question"]}

                q_id = elem.get("question_id")

                if q_id:
                    question_data["question_id"] = q_id
                list_questions.append(UnansweredQuestion(**question_data))

            minimal_results: list[MinimalSearchResults] = []
            for quest in tqdm(list_questions, desc="recherche des sources"):
                src = self.rt.search_query(quest.question, k,
                                           quest.question_id)
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

    def answer(self,
               question: str,
               k: int = 10,
               question_id: str | None = None) -> None:
        try:
            result = self.rt.answer_query(question, k, question_id)
            print(result.model_dump_json(indent=2))
            with open("data/processed/answer.json", "w",
                      encoding="utf-8") as r:
                r.write(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Erreur à la question : {e}")

    def answer_dataset(self,
                       student_search_results_path: str | Path,
                       save_directory: str | Path = (
                           'data/output/search_results_and_answer/'
                       ),
                       k: int = 10) -> None:

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

    def evaluate(self,
                 student_answer_path: str | Path,
                 dataset_path: str | Path,
                 k: int = 10,
                 max_context_length: int = 2000) -> None:
        """Évalue les résultats de recherche contre le Ground Truth."""
        from src.evaluator import Evaluator

        ev = Evaluator(
            student_answer_path=Path(student_answer_path),
            dataset_path=Path(dataset_path)
        )
        ev.run_evaluation(k=k)


def main() -> None:
    fire.Fire(RagCLI)


if __name__ == '__main__':
    main()
