"""FastAPI application for the RVCE Campus Information Assistant."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.answering import NO_EVIDENCE_MESSAGE, answer_question
from src.generation import CLEAN_GENERATION_ERROR_MESSAGE
from src.gmail_ingestion import GmailSyncWorker
from src.mcp_server import CampusGmailMCPClient, CampusGmailMCPServer

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_TOKEN_PATH = BASE_DIR / "secrets" / "token.json"

gmail_sync_worker = GmailSyncWorker(
    interval_seconds=int(os.environ.get("GMAIL_SYNC_INTERVAL_SECONDS", "300")),
)
mcp_server = CampusGmailMCPServer()
mcp_client = CampusGmailMCPClient(server=mcp_server)


def _sanitize_public_error_payload(
    answer: str,
    error: Optional[Dict[str, Any]],
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Ensure public API responses never expose raw backend/provider errors or keys."""
    if not error:
        return answer, None

    category = str(error.get("category") or "generation_unavailable")
    code = str(error.get("code") or error.get("error_code") or category)
    retryable = bool(error.get("retryable", False))

    safe_error = {
        "category": category,
        "code": code,
        "error_code": code,
        "message": CLEAN_GENERATION_ERROR_MESSAGE,
        "retryable": retryable,
    }
    return CLEAN_GENERATION_ERROR_MESSAGE, safe_error


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    auto_sync_env = os.environ.get("GMAIL_AUTO_SYNC", "").strip().lower()
    is_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    should_start_worker = auto_sync_env in {"1", "true", "yes"} or (
        auto_sync_env not in {"0", "false", "no"}
        and DEFAULT_TOKEN_PATH.exists()
        and not is_pytest
    )
    if should_start_worker:
        gmail_sync_worker.start()
    try:
        yield
    finally:
        if gmail_sync_worker.is_running:
            gmail_sync_worker.stop(timeout=2.0)


app = FastAPI(title="RVCE Campus Information Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The student question to answer.")
    strategy: Optional[str] = Field(
        "hybrid",
        description="Retrieval strategy: 'hybrid' (default), 'semantic', 'tfidf', 'hybrid_rerank', or 'gpt' (OpenAI GPT answer generation).",
    )
    rerank: Optional[bool] = Field(False, description="Enable cross-encoder reranking if available.")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional prior conversation turns [{'role': 'user'|'assistant', 'content': '...'}] for follow-up continuity.",
    )


class QuestionResponse(BaseModel):
    answer: str
    abstention_status: bool
    sources: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    strategy: Optional[str] = "hybrid"
    intent: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


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

    strategy = payload.strategy or "hybrid"
    rerank = payload.rerank or False
    try:
        result = answer_question(
            question,
            strategy=strategy,
            rerank=rerank,
            conversation_history=payload.conversation_history,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unhandled error in ask_question")
        result = {
            "answer": CLEAN_GENERATION_ERROR_MESSAGE,
            "evidence": [],
            "sources": [],
            "strategy": strategy,
            "error": {
                "category": "internal_error",
                "code": "generation_unavailable",
                "error_code": "generation_unavailable",
                "message": CLEAN_GENERATION_ERROR_MESSAGE,
                "retryable": True,
            },
        }

    answer = result.get("answer") or NO_EVIDENCE_MESSAGE
    evidence = result.get("evidence") or []
    sources = result.get("sources") or []
    error = result.get("error")
    answer, error = _sanitize_public_error_payload(answer, error)

    effective_strategy = result.get("strategy") or strategy
    if rerank and "rerank" not in effective_strategy:
        effective_strategy = f"{effective_strategy}+rerank"

    return {
        "answer": answer,
        "abstention_status": answer == NO_EVIDENCE_MESSAGE,
        "sources": sources,
        "evidence": evidence,
        "strategy": effective_strategy,
        "intent": result.get("intent"),
        "error": error,
    }


@app.get("/sync/status")
def sync_status() -> Dict[str, Any]:
    """Return current Gmail synchronization status, corpus statistics, and MCP health."""
    status = gmail_sync_worker.get_status()
    status["mcp"] = mcp_client.get_status()
    return status


@app.post("/sync/trigger")
def trigger_sync(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Trigger an asynchronous incremental Gmail synchronization pass without blocking."""
    background_tasks.add_task(gmail_sync_worker.sync_once)
    return {
        "status": "scheduled",
        "message": "Incremental Gmail synchronization scheduled in background.",
    }


@app.get("/mcp/status")
def get_mcp_status() -> Dict[str, Any]:
    """Verify and return operational status of the Gmail & Campus Retrieval MCP server."""
    if not mcp_client.connected:
        mcp_client.connect()
    return mcp_client.get_status()
