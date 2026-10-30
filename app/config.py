# File: retail-rag-agent/app/config.py
"""
Central configuration (12-factor style).

Every field can be overridden by an environment variable with the same name,
case-insensitive. Example: GROQ_MODEL=qwen/qwen3.8-27b
Secrets (API keys) come ONLY from the environment / .env file, never from code.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "NovaMart Assistant"
    environment: str = "dev"
    log_level: str = "INFO"

    # --- LLM providers --------------------------------------------------
    # Groq is REQUIRED: it powers answers, query rewriting and the agents.
    # Gemini is OPTIONAL: if its key is set, it becomes the fallback when Groq
    # fails and an independent judge for evaluation.
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"
    # Qwen 3.8 can "think" before answering. "none" turns that off: faster replies
    # and fewer tokens, which matters on the free tier (8K tokens per minute).
    groq_reasoning_effort: str = "none"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 4  # the SDK waits and retries on 429 (rate limit) and 5xx errors

    # --- Retrieval models (run locally on CPU via ONNX, no API cost) ----
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    model_cache_dir: Path = BASE_DIR / "models"

    # --- Paths ------------------------------------------------------------
    data_dir: Path = BASE_DIR / "data" / "raw"
    artifacts_dir: Path = BASE_DIR / "artifacts"

    # --- Chunking ---------------------------------------------------------
    chunk_max_chars: int = 900
    chunk_overlap_chars: int = 150

    # --- Retrieval pipeline ----------------------------------------------
    # Defaults below are a starting point. Run `python -m eval.run_eval retrieval`:
    # it tests many combinations and prints the best values to paste into .env.
    dense_top_k: int = 30  # candidates from FAISS
    sparse_top_k: int = 30  # candidates from BM25
    retrieval_dense_weight: float = 1.0  # weight of meaning search in fusion (0 = off)
    retrieval_sparse_weight: float = 0.3  # weight of keyword search in fusion (0 = off)
    rrf_k: int = 60  # RRF constant: smaller = top ranks count more
    retrieval_rerank: bool = False  # cross-encoder second stage; costs ~0.7 s per query on a laptop CPU
    # BGE models were trained with this instruction in front of search queries (not documents).
    embedding_query_prefix: str = ""
    rerank_candidates: int = 20  # how many fused candidates the cross-encoder re-scores
    final_top_k: int = 5  # chunks sent to the LLM
    # If set, answers are refused when the best rerank score is below this.
    # Calibrate it with `python -m eval.run_eval retrieval` (see README).
    min_rerank_score: float | None = None

    # --- Answering ----------------------------------------------------------
    # Query planning: one fast LLM call turns the message into 1-3 search queries
    # (one per sub-question, with follow-up references resolved). Multi-part
    # questions like "features AND max discount" then get evidence for every part.
    query_planning: bool = True
    max_search_queries: int = 3
    context_chunks: int = 6  # chunks sent to the LLM after merging all queries' results
    # Shown whenever the assistant cannot answer, so the customer always has a next step.
    support_contact: str = "NovaMart Support: chat on the Help page or call 1800-123-4567 (9 AM to 9 PM daily)"

    # --- Conversation memory ---------------------------------------------
    max_history_turns: int = 4
    session_ttl_s: int = 3600

    # --- Agent mode (our own agent loop, see app/agents/) ------------------
    agent_max_steps: int = 4  # max tool-calling rounds per agent before it must answer
    agent_tool_output_chars: int = 2500  # tool results are truncated to protect the token budget
    agent_temperature: float = 0.3
    # Run the two research agents at the same time. Faster on a paid tier; on the
    # free tier both compete for the same 8K tokens/minute, so it is off by default.
    agent_parallel: bool = False


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed once per process."""
    return Settings()