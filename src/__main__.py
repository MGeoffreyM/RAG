import json
import fire
from pathlib import Path
from tqdm import tqdm
from src.chunking import Chunker
from src.retriever import Retriever
from src.Llm import Llm
from src.tools import (UnansweredQuestion,
                       StudentSearchResults,
                       MinimalSearchResults)


class RagCLI:
    """ Interface en ligne de commande"""
    def __init__(self) -> None:
        self.index_path = Path("data/processed/bm25_index")
        self.chunks_path = Path("data/processed/chunks/chunks.json")
        self.rt = Retriever(
            index_dir=self.index_path, chunks_file=self.chunks_path)
        self.llm = Llm("Qwen/Qwen3-0.6B")

    def index(self,
              data_path: Path = Path(
                  '/home/gematura/cursus_42/rag/data/vllm-0.10.1/'),
              max_chunk_size: int = 2000,
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
            src = self.rt.search_query(quest.question, k, quest.question_id)
            minimal_results.append(src.search_results[0])

        final_results = StudentSearchResults(
            search_results=minimal_results,
            k=k
        )

        save_directory.mkdir(parents=True, exist_ok=True)
        file_path = save_directory / dataset_path.name
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(final_results.model_dump_json(indent=2))

    def answer(self,
               question: str,
               k: int = 10,
               question_id: str | None = None) -> None:
        try:
            result = self.rt.answer_query_query(question, k, question_id)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Erreur à la recherche : {e}")

    def answer_dataset(self) -> None:
        print("answer_dataset a implementer")

    def evaluate(self) -> None:
        print("evaluate a implementer")


def main() -> None:
    fire.Fire(RagCLI)


if __name__ == '__main__':
    main()
