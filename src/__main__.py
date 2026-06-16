import fire
from pathlib import Path
from src.chunking import Chunker


class RagCLI:
    """ Interface en ligne de commande"""
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
            print(f"Indexation terminée ! {len(ch.list_chunk)} chunks créés.")

        except Exception as e:
            print(f"Erreur à l'indexation : {e}")

    def search(self) -> None:
        print("search a implementer")

    def search_dataset(self) -> None:
        print("search_dataset a implementer")

    def answer(self) -> None:
        print("answer a implementer")

    def answer_dataset(self) -> None:
        print("answer_dataset a implementer")

    def evaluate(self) -> None:
        print("evaluate a implementer")


def main() -> None:
    fire.Fire(RagCLI)


if __name__ == '__main__':
    main()
