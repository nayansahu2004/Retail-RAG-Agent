import pytest

from app.llm.client import FallbackLLM, LLMError, strip_reasoning


class Provider:
    def __init__(self, name, reply=None, error=None):
        self.name, self.model, self.reply, self.error = name, f"{name}-model", reply, error

    def generate(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.reply


def test_falls_back_to_second_provider():
    llm = FallbackLLM([Provider("groq", error=TimeoutError()), Provider("gemini", reply="hi")])
    assert llm.generate("sys", []).label == "gemini/gemini-model"


def test_raises_when_all_fail_or_none_configured():
    with pytest.raises(LLMError):
        FallbackLLM([Provider("groq", reply="  ")]).generate("sys", [])
    with pytest.raises(LLMError):
        FallbackLLM([])


def test_strip_reasoning():
    assert strip_reasoning("<think>hmm</think> Answer [1]") == "Answer [1]"


def test_groq_provider_sends_tools_and_parses_tool_calls(settings):
    """Verify the request body sent to Groq and the parsing of tool-call replies, using a mock transport."""
    import json

    import httpx
    from groq import Groq

    from app.llm.client import GroqProvider

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": captured["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "web_search", "arguments": '{"query": "ram"}'},
                                }
                            ],
                        },
                    }
                ],
            },
        )

    settings.groq_api_key = "test"
    provider = GroqProvider(settings)
    provider._client = Groq(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    tools = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}]
    turn = provider.chat([{"role": "user", "content": "hi"}], tools=tools)

    assert captured["model"] == "qwen/qwen3.8-27b"
    assert captured["reasoning_effort"] == "none" and captured["reasoning_format"] == "hidden"
    assert captured["tools"] == tools and captured["tool_choice"] == "auto"
    assert turn.tool_calls[0].name == "web_search" and turn.tool_calls[0].arguments == '{"query": "ram"}'
