# File: retail-rag-agent/app/api/routes.py
"""
HTTP endpoints. Kept thin: validate input, call a service, shape the output.

Endpoints are plain `def` (not `async def`) on purpose: the LLM SDK, FAISS and
ONNX calls are blocking. FastAPI runs `def` endpoints in a thread pool, so one
slow request does not freeze the event loop for everyone else.
"""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from app.config import get_settings
from app.llm.client import LLMError
from app.observability import AGENT_RUNS
from app.rag.prompts import format_history
from app.schemas import AgentStep, ChatRequest, ChatResponse, Source, ToolCallTrace
from app.services import Services

logger = logging.getLogger(__name__)
router = APIRouter()


def get_services(request: Request) -> Services:
    return request.app.state.services


# Annotated dependency: FastAPI's recommended style, reusable across endpoints.
ServicesDep = Annotated[Services, Depends(get_services)]


@router.get("/health", tags=["ops"])
def health() -> dict:
    """Liveness: the process is up. Used by Docker HEALTHCHECK."""
    return {"status": "ok"}


@router.get("/ready", tags=["ops"])
def ready(services: ServicesDep) -> dict:
    """Readiness: the index is loaded and the app can answer."""
    return {
        "status": "ready",
        "chunks": len(services.retriever.chunks),
        "agent_mode": services.team is not None,
    }


@router.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"])
def chat(req: ChatRequest, services: ServicesDep) -> ChatResponse:
    session_id = req.session_id or uuid.uuid4().hex
    if req.mode == "agent":
        return _agent_answer(req, session_id, services)

    try:
        result = services.rag.answer(req.question, session_id)
    except LLMError as exc:
        logger.error("all LLM providers failed", extra={"extra_fields": {"error": str(exc)[:300]}})
        raise HTTPException(
            status_code=503, detail="The answer service is busy. Please try again in a moment."
        ) from exc

    sources = [
        Source(
            ref=i,
            chunk_id=r.chunk.id,
            title=r.chunk.title,
            source_type=r.chunk.source_type,
            score=round(r.rerank_score if r.rerank_score is not None else r.score, 4),
            cited=i in result.cited,
            snippet=r.chunk.text[:280],
        )
        for i, r in enumerate(result.results, start=1)
    ]
    return ChatResponse(
        answer=result.answer,
        mode="rag",
        session_id=session_id,
        search_queries=result.search_queries,
        coverage=result.coverage,
        sources=sources,
        timings_ms=result.timings_ms,
        llm=result.llm,
    )


def _agent_answer(req: ChatRequest, session_id: str, services: Services) -> ChatResponse:
    if services.team is None:
        raise HTTPException(status_code=503, detail="Research mode is turned off: the server has no GROQ_API_KEY.")
    history = services.sessions.get(session_id)
    try:
        result = services.team.run(req.question, format_history(history))
    except Exception as exc:
        AGENT_RUNS.labels(outcome="error").inc()
        logger.exception("agent run failed")
        raise HTTPException(status_code=502, detail="Research mode failed. Try Quick answer, or retry.") from exc
    AGENT_RUNS.labels(outcome="ok").inc()
    services.sessions.append(session_id, req.question, result.answer)

    steps = [
        AgentStep(
            agent=step.agent,
            output=step.output,
            tool_calls=[ToolCallTrace(tool=t.tool, arguments=t.arguments, ok=t.ok, ms=t.ms) for t in step.tool_calls],
            llm_calls=step.llm_calls,
            ms=step.ms,
        )
        for step in result.steps
    ]
    return ChatResponse(
        answer=result.answer,
        mode="agent",
        session_id=session_id,
        agent_steps=steps,
        timings_ms={"total_ms": result.total_ms},
        llm=f"groq/{get_settings().groq_model}",
    )