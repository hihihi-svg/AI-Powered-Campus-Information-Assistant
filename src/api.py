"""Minimal FastAPI application for the RVCE Campus Information Assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.answering import NO_EVIDENCE_MESSAGE, answer_question

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="RVCE Campus Information Assistant")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The student question to answer.")
    strategy: Optional[str] = Field("tfidf", description="Retrieval strategy: 'tfidf' (default), 'semantic', 'hybrid', or 'hybrid_rerank'.")
    rerank: Optional[bool] = Field(False, description="Enable cross-encoder reranking if available.")


class QuestionResponse(BaseModel):
    answer: str
    abstention_status: bool
    sources: List[Dict[str, Optional[str]]]
    evidence: List[Dict[str, Any]]
    strategy: Optional[str] = "tfidf"


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "rvce-campus-information-assistant"}


@app.post("/ask", response_model=QuestionResponse)
def ask_question(payload: QuestionRequest) -> Dict[str, Any]:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must be a non-empty string.")

    strategy = payload.strategy or "tfidf"
    rerank = payload.rerank or False
    result = answer_question(question, strategy=strategy, rerank=rerank)
    answer = result.get("answer") or NO_EVIDENCE_MESSAGE
    evidence = result.get("evidence") or []
    sources = result.get("sources") or []

    effective_strategy = strategy
    if rerank and "rerank" not in strategy:
        effective_strategy = f"{strategy}+rerank"

    return {
        "answer": answer,
        "abstention_status": answer == NO_EVIDENCE_MESSAGE,
        "sources": sources,
        "evidence": evidence,
        "strategy": effective_strategy,
    }

