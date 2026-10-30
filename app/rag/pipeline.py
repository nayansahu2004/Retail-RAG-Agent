# File: retail-rag-agent/app/rag/pipeline.py
"""
The full RAG answer flow (Quick answer mode):

1. Load chat history for the session.
2. PLAN: one fast LLM call turns the message into 1-3 standalone search queries.
   - Follow-ups get their references resolved ("is it waterproof?" ->
     "Orbit Ultra 512 waterproof").
   - Multi-part questions get one query per part ("features and max discount"
     -> a product query AND an offers query). A single query would let one part
     crowd out the evidence for the other.
3. RETRIEVE for every query, then MERGE round-robin (best hit of each query
   first), so every part of the question is represented in the context.
4. GUARDRAIL: if even the best chunk is a poor match, answer "not found" with
   the support contact, without calling the LLM.
5. GENERATE: grounded JSON answer {"answer", "coverage"} with [n] citations.
   Coverage (full / partial / none) is reported by the model, so the code does
   not have to guess from the wording whether the bot answered.
6. Parse citations, save the turn, return timings.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.llm.client import ChatLLM, LLMError
from app.memory import SessionStore
from app.observability import REFUSALS, STAGE_LATENCY
from app.rag.entities import RetrievedChunk
from app.rag.prompts import (
    COVERAGE_VALUES,
    PLAN_SYSTEM_PROMPT,
    answer_system_prompt,
    format_history,
    format_sources,
    not_found_message,
)
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)
_CITATION = re.compile(r"\[(\d+)\]")
_CODE_FENCE = re.compile(r"^```(?:json)?|```$", re.M)


@dataclass
class RAGAnswer:
    answer: str
    search_queries: list[str]
    results: list[RetrievedChunk]
    cited: set[int]
    coverage: str = "full"
    timings_ms: dict[str, float] = field(default_factory=dict)
    llm: str | None = None

    @property
    def refused(self) -> bool:
        return self.coverage == "none"


def extract_citations(answer: str, n_sources: int) -> set[int]:
    """Valid [n] citations only; a hallucinated [9] with 5 sources is ignored."""
    return {int(n) for n in _CITATION.findall(answer) if 1 <= int(n) <= n_sources}


def _load_json(text: str) -> dict | None:
    try:
        data = json.loads(_CODE_FENCE.sub("", text.strip()).strip())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_answer(text: str) -> tuple[str, str]:
    """Return (answer, coverage). If the model ignored JSON format, keep its text as a full answer."""
    data = _load_json(text)
    if not data or not isinstance(data.get("answer"), str) or not data["answer"].strip():
        return text.strip(), "full"
    coverage = data.get("coverage") if data.get("coverage") in COVERAGE_VALUES else "full"
    return data["answer"].strip(), coverage


def parse_queries(text: str, fallback: str, max_queries: int) -> list[str]:
    data = _load_json(text)
    raw = data.get("queries") if data else None
    if not isinstance(raw, list):
        return [fallback]
    queries: list[str] = []
    for q in raw:
        if isinstance(q, str) and 2 <= len(q.strip()) <= 300 and q.strip() not in queries:
            queries.append(q.strip())
    return queries[:max_queries] or [fallback]


def merge_round_robin(result_lists: list[list[RetrievedChunk]], limit: int) -> list[RetrievedChunk]:
    """Take the best hit of each query, then the second best of each, ... skipping duplicates."""
    merged: list[RetrievedChunk] = []
    seen: set[str] = set()
    for position in range(max((len(r) for r in result_lists), default=0)):
        for results in result_lists:
            if position < len(results) and results[position].chunk.id not in seen:
                seen.add(results[position].chunk.id)
                merged.append(results[position])
                if len(merged) == limit:
                    return merged
    return merged


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: ChatLLM, sessions: SessionStore, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.sessions = sessions
        self.settings = settings

    def plan(self, question: str, history: list[dict]) -> list[str]:
        prompt = f"Conversation:\n{format_history(history) or 'none'}\n\nLatest message: {question}"
        try:
            result = self.llm.generate(
                PLAN_SYSTEM_PROMPT, [{"role": "user", "content": prompt}], temperature=0.0, json_mode=True
            )
        except LLMError:
            logger.warning("Query planning failed; searching with the raw question")
            return [question]  # graceful degradation: a worse search beats a failed request
        return parse_queries(result.text, question, self.settings.max_search_queries)

    def answer(self, question: str, session_id: str) -> RAGAnswer:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        history = self.sessions.get(session_id)
        s = self.settings

        queries = [question]
        if s.query_planning or history:
            t = time.perf_counter()
            queries = self.plan(question, history)
            seconds = time.perf_counter() - t
            STAGE_LATENCY.labels(stage="plan").observe(seconds)
            timings["plan_ms"] = round(seconds * 1000, 2)

        result_lists = []
        for query in queries:
            results, retrieval_timings = self.retriever.retrieve(query)
            result_lists.append(results)
            for key, value in retrieval_timings.items():  # sum stage times across queries
                timings[key] = round(timings.get(key, 0.0) + value, 2)
        results = merge_round_robin(result_lists, s.context_chunks)

        llm_label = None
        if self._should_refuse(results):
            answer, coverage = not_found_message(s.support_contact), "none"
        else:
            user_message = (
                f"<sources>\n{format_sources(results)}\n</sources>\n\n"
                f"Customer question: {question}\n"
                f"(Searches used: {' | '.join(queries)})"
            )
            result = self.llm.generate(
                answer_system_prompt(s.support_contact),
                [*history, {"role": "user", "content": user_message}],
                json_mode=True,
            )
            answer, coverage = parse_answer(result.text)
            llm_label = result.label
            timings["generate_ms"] = result.latency_ms

        if coverage == "none":
            REFUSALS.inc()
        response = RAGAnswer(
            answer=answer,
            search_queries=queries,
            results=results,
            cited=extract_citations(answer, len(results)),
            coverage=coverage,
            timings_ms=timings,
            llm=llm_label,
        )
        self.sessions.append(session_id, question, answer)
        response.timings_ms["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "rag answer",
            extra={
                "extra_fields": {
                    "queries": len(queries),
                    "cited": sorted(response.cited),
                    "coverage": coverage,
                    **response.timings_ms,
                }
            },
        )
        return response

    def _should_refuse(self, results: list[RetrievedChunk]) -> bool:
        if not results:
            return True
        threshold = self.settings.min_rerank_score
        top = results[0].rerank_score
        return threshold is not None and top is not None and top < threshold