# File: retail-rag-agent/app/observability.py
"""
Custom Prometheus metrics (exposed at GET /metrics).

HTTP metrics (request count, latency per route) come from
prometheus-fastapi-instrumentator. These add RAG-specific signals that tell you
WHERE time goes and how often the primary LLM fails.
"""

from prometheus_client import Counter, Histogram

STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds",
    "Latency of each RAG pipeline stage",
    ["stage"],  # plan | dense | sparse | rerank | generate | agent_llm
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

LLM_CALLS = Counter("llm_calls_total", "LLM calls by provider and outcome", ["provider", "outcome"])

REFUSALS = Counter("rag_refusals_total", "Answers where the bot could not answer any part of the question")

AGENT_RUNS = Counter("agent_runs_total", "Agent-mode runs by outcome", ["outcome"])