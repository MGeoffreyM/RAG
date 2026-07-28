from pathlib import Path
from fastapi import FastAPI, Query, Depends
from fastapi.responses import RedirectResponse
from src.retriever import Retriever
from src.cli import RagCLI
from src.tools import (UnansweredQuestion,
                       StudentSearchResults,
                       StudentSearchResultsAndAnswer)


app = FastAPI(title="RAG API")


rt = Retriever(
    index_dir=Path("data/processed/bm25_index"),
    chunks_file=Path("data/processed/chunks/chunks.json"),
    semantic=False
)


def verify_index(
    use_semantic: bool = Query(False,
                               description="Activer la recherche sémantique")
) -> bool:
    """
    Dépendance exécutée AVANT chaque requête.
    Vérifie si l'index existe et lance l'indexation si nécessaire.
    """
    rt.semantic = use_semantic

    index_missing = not rt.chunks_file.exists()

    chroma_dir = rt.index_dir / "chroma_db"
    semantic_missing = use_semantic and not chroma_dir.exists()

    if index_missing or semantic_missing:
        print("Index manquant. Lancement de l'indexation...")
        cli = RagCLI(semantic=use_semantic)
        cli.index()

    return use_semantic


@app.post("/search")
def api_search(
    query: str,
    k: int = 10,
    use_semantic: bool = Depends(verify_index)
) -> StudentSearchResults:
    """Endpoint pour chercher les sources."""

    rt.semantic = use_semantic

    ques = UnansweredQuestion(question=query)
    return rt.search_query(ques, k)


@app.post("/answer")
def api_answer(
    query: str,
    k: int = 10,
    use_semantic: bool = Depends(verify_index)
) -> StudentSearchResultsAndAnswer:
    """Endpoint pour générer une réponse."""

    rt.semantic = use_semantic

    ques = UnansweredQuestion(question=query)
    return rt.answer_query(ques, k)


@app.get("/", include_in_schema=False)
def root_redirect() -> 'RedirectResponse':
    """Redirige la racine vers la documentation de l'API (Swagger)."""
    return RedirectResponse(url="/docs")
