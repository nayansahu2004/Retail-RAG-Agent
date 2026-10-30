"""LLM client interfaces, provider implementations, and a failover wrapper.

The module exposes two capabilities:

* :class:`ChatLLM.generate` for plain text completion, used by the RAG pipeline
  and the evaluation judge. Requests are routed through :class:`FallbackLLM`,
  which tries each configured provider in priority order.
* :class:`ToolCallingLLM.chat` for the OpenAI-compatible tool-calling API that
  drives the research agents.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from app.config import Settings
from app.observability import LLM_CALLS, STAGE_LATENCY

logger = logging.getLogger(__name__)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


class LLMError(RuntimeError):
    """Raised when every configured LLM provider has failed for a single call."""


@dataclass
class LLMResult:
    """Successful completion returned by :class:`FallbackLLM`."""

    text: str
    provider: str
    model: str
    latency_ms: float

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: str


@dataclass
class AssistantTurn:
    """One assistant turn in tool-calling mode (text, tool calls, or both)."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatLLM(Protocol):
    """Protocol satisfied by any text-generation backend."""

    def generate(
        self, system: str, messages: list[dict], temperature: float | None = None, json_mode: bool = False
    ) -> LLMResult: ...


class ToolCallingLLM(Protocol):
    """Protocol satisfied by any tool-calling backend."""

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> AssistantTurn: ...


def strip_reasoning(text: str) -> str:
    """Strip inline ``<think>...</think>`` reasoning blocks from ``text``."""
    return _THINK_BLOCK.sub("", text).strip()


class GroqProvider:
    """Groq-backed implementation of both :class:`ChatLLM` and :class:`ToolCallingLLM`."""

    name = "groq"

    def __init__(self, settings: Settings, temperature: float | None = None):
        from groq import Groq

        self.model = settings.groq_model
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.max_tokens = settings.llm_max_tokens
        self.reasoning_effort = settings.groq_reasoning_effort
        self._client = Groq(
            api_key=settings.groq_api_key, timeout=settings.llm_timeout_s, max_retries=settings.llm_max_retries
        )

    def _model_params(self) -> dict:
        if self.model.startswith("qwen/") and self.reasoning_effort:
            return {"reasoning_effort": self.reasoning_effort, "reasoning_format": "hidden"}
        return {}

    def _complete(self, messages: list[dict], temperature: float | None, **extra):
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_completion_tokens=self.max_tokens,
            **self._model_params(),
            **extra,
        )

    def generate(self, system, messages, temperature=None, json_mode=False) -> str:
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = self._complete([{"role": "system", "content": system}, *messages], temperature, **extra)
        return strip_reasoning(response.choices[0].message.content or "")

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> AssistantTurn:
        started = time.perf_counter()
        extra = {"tools": tools, "tool_choice": "auto"} if tools else {}
        try:
            response = self._complete(messages, None, **extra)
        except Exception:
            LLM_CALLS.labels(provider=self.name, outcome="error").inc()
            raise
        LLM_CALLS.labels(provider=self.name, outcome="ok").inc()
        STAGE_LATENCY.labels(stage="agent_llm").observe(time.perf_counter() - started)
        message = response.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "{}")
            for c in (message.tool_calls or [])
        ]
        return AssistantTurn(content=strip_reasoning(message.content or ""), tool_calls=calls)


class GeminiProvider:
    """Gemini-backed implementation of :class:`ChatLLM`."""

    name = "gemini"

    def __init__(self, settings: Settings):
        from google import genai
        from google.genai import types

        self._types = types
        self.model = settings.gemini_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self._client = genai.Client(
            api_key=settings.gemini_api_key, http_options=types.HttpOptions(timeout=int(settings.llm_timeout_s * 1000))
        )

    def generate(self, system, messages, temperature=None, json_mode=False) -> str:
        types = self._types
        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        response = self._client.models.generate_content(model=self.model, contents=contents, config=config)
        return response.text or ""


class FallbackLLM:
    """Wrap several providers and try them in order until one succeeds."""

    def __init__(self, providers: list):
        if not providers:
            raise LLMError("No LLM provider configured. Set GROQ_API_KEY.")
        self.providers = providers

    def generate(self, system, messages, temperature=None, json_mode=False) -> LLMResult:
        errors: list[str] = []
        for provider in self.providers:
            started = time.perf_counter()
            try:
                text = provider.generate(system, messages, temperature=temperature, json_mode=json_mode)
                if not text.strip():
                    raise LLMError("empty response")
            except Exception as exc:
                LLM_CALLS.labels(provider=provider.name, outcome="error").inc()
                logger.warning(
                    "LLM provider failed, trying next",
                    extra={"extra_fields": {"provider": provider.name, "error": repr(exc)[:300]}},
                )
                errors.append(f"{provider.name}: {exc!r}"[:300])
                continue
            seconds = time.perf_counter() - started
            LLM_CALLS.labels(provider=provider.name, outcome="ok").inc()
            STAGE_LATENCY.labels(stage="generate").observe(seconds)
            return LLMResult(text.strip(), provider.name, provider.model, round(seconds * 1000, 2))
        raise LLMError("All LLM providers failed: " + " | ".join(errors))


def build_llm(settings: Settings, order: tuple[str, ...] = ("groq", "gemini")) -> FallbackLLM:
    """Instantiate every configured provider in priority ``order``."""
    factories = {
        "groq": (settings.groq_api_key, lambda: GroqProvider(settings)),
        "gemini": (settings.gemini_api_key, lambda: GeminiProvider(settings)),
    }
    return FallbackLLM([factories[name][1]() for name in order if factories[name][0]])
