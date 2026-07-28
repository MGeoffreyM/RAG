import fire
from src.cli import RagCLI


def main() -> None:
    try:
        fire.Fire(RagCLI)
    except Exception as e:
        print(f"ERREUR: {e}")


if __name__ == '__main__':
    main()
