"""Dependency container built once at startup and shared across requests."""

import logging
from dataclasses import dataclass

from app.agents.team import ShoppingResearchTeam
from app.config import Settings
from app.memory import SessionStore
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class Services:
    """Holds the long-lived components wired into the FastAPI app state."""

    retriever: HybridRetriever
    rag: RAGPipeline
    sessions: SessionStore
    team: ShoppingResearchTeam | None = None


def build_services(settings: Settings) -> Services:
    """Construct the production service container from settings.

    Args:
        settings: Fully-resolved application configuration.

    Returns:
        A ``Services`` instance with the retriever, RAG pipeline, session
        store, and (when ``GROQ_API_KEY`` is configured) the agent team.
    """
    from app.llm.client import GroqProvider, build_llm
    from app.rag.retriever import build_retriever

    retriever = build_retriever(settings)
    sessions = SessionStore(settings.max_history_turns, settings.session_ttl_s)
    rag = RAGPipeline(retriever, build_llm(settings, order=("groq", "gemini")), sessions, settings)

    team = None
    if settings.groq_api_key:
        agent_llm = GroqProvider(settings, temperature=settings.agent_temperature)
        team = ShoppingResearchTeam(retriever, agent_llm, settings)
    else:
        logger.warning("GROQ_API_KEY not set: research mode disabled")

    logger.info(
        "services ready",
        extra={"extra_fields": {"chunks": len(retriever.chunks), "agent_mode": team is not None}},
    )
    return Services(retriever=retriever, rag=rag, sessions=sessions, team=team)
