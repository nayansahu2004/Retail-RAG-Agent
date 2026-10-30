# File: retail-rag-agent/app/schemas.py
"""API request/response contracts. FastAPI validates and documents these automatically (/docs)."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000, examples=["What is the return window for shoes?"])
    session_id: str | None = Field(default=None, max_length=64, description="Reuse to keep conversation context")
    mode: Literal["rag", "agent"] = "rag"


class Source(BaseModel):
    ref: int = Field(description="The [n] number the answer uses to cite this source")
    chunk_id: str
    title: str
    source_type: Literal["product", "policy"]
    score: float
    cited: bool
    snippet: str


class ToolCallTrace(BaseModel):
    tool: str
    arguments: str
    ok: bool
    ms: float


class AgentStep(BaseModel):
    agent: str
    output: str
    tool_calls: list[ToolCallTrace] = []
    llm_calls: int = 0
    ms: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["rag", "agent"]
    session_id: str
    search_queries: list[str] = []
    coverage: Literal["full", "partial", "none"] | None = None
    sources: list[Source] = []
    agent_steps: list[AgentStep] = []
    timings_ms: dict[str, float] = {}
    llm: str | None = None