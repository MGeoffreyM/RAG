from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.retriever import Retriever
from src.tools import (UnansweredQuestion,
                       StudentSearchResults,
                       StudentSearchResultsAndAnswer)


app = FastAPI(title="RAG API")


rt = Retriever(
    index_dir=Path("data/processed/bm25_index"),
    chunks_file=Path("data/processed/chunks/chunks.json")
)


@app.post("/search")
def api_search(query: str, k: int = 10) -> StudentSearchResults:
    """Endpoint pour chercher les sources"""
    ques = UnansweredQuestion(question=query)
    return rt.search_query(ques, k)


@app.post("/answer")
def api_answer(query: str, k: int = 10) -> StudentSearchResultsAndAnswer:
    """Endpoint pour générer une réponse"""
    ques = UnansweredQuestion(question=query)
    return rt.answer_query(ques, k)


@app.get("/", include_in_schema=False)
def root_redirect() -> 'RedirectResponse':
    """Redirige la racine vers la documentation de l'API."""
    return RedirectResponse(url="/docs")
