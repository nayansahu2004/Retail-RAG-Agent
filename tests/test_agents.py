"""Agent-loop tests using a scripted fake LLM for deterministic behaviour."""

import json

from app.agents.agent import FINAL_ANSWER_NUDGE, Agent
from app.agents.team import ShoppingResearchTeam
from app.agents.tools import make_catalog_tool, make_web_search_tool
from app.llm.client import AssistantTurn, ToolCall


class ScriptedLLM:
    """Return pre-written assistant turns in order and record each request."""

    def __init__(self, turns: list[AssistantTurn]):
        self.turns = list(turns)
        self.requests: list[dict] = []

    def chat(self, messages, tools=None):
        self.requests.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.turns.pop(0) if self.turns else AssistantTurn(content="done")


def call(name: str, **arguments) -> AssistantTurn:
    return AssistantTurn(
        content="", tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=json.dumps(arguments))]
    )


def test_tool_schema_comes_from_pydantic(retriever):
    schema = make_catalog_tool(retriever).schema()["function"]
    assert schema["name"] == "search_product_catalog"
    assert "query" in schema["parameters"]["required"]
    assert "Laptops" in schema["description"]


def test_catalog_tool_applies_price_and_category_filters(retriever):
    output, ok = make_catalog_tool(retriever).run(
        json.dumps({"query": "Kestrel laptop", "max_price_inr": 60000, "category": "Laptops"})
    )
    assert ok and "SKU-LAP-001" in output and "SKU-LAP-002" not in output


def test_invalid_arguments_are_reported_not_raised(retriever):
    output, ok = make_catalog_tool(retriever).run('{"max_price_inr": "cheap"}')
    assert not ok and "Invalid arguments" in output


def test_agent_calls_tool_then_answers(retriever):
    llm = ScriptedLLM(
        [call("search_product_catalog", query="silent laptop"), AssistantTurn(content="Pick SKU-LAP-001")]
    )
    result = Agent("Catalog", "instructions", [make_catalog_tool(retriever)], llm).run("find a silent laptop")

    assert result.output == "Pick SKU-LAP-001"
    assert [t.tool for t in result.tool_calls] == ["search_product_catalog"] and result.tool_calls[0].ok
    tool_message = llm.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call_search_product_catalog"


def test_unknown_tool_is_reported_to_the_model(retriever):
    llm = ScriptedLLM([call("delete_database"), AssistantTurn(content="ok")])
    result = Agent("A", "i", [make_catalog_tool(retriever)], llm).run("task")
    assert not result.tool_calls[0].ok
    assert "Unknown tool" in llm.requests[1]["messages"][-1]["content"]


def test_step_budget_forces_final_answer_without_tools(retriever):
    llm = ScriptedLLM([call("search_product_catalog", query="x")] * 2 + [AssistantTurn(content="final")])
    result = Agent("A", "i", [make_catalog_tool(retriever)], llm, max_steps=2).run("task")
    assert result.output == "final" and result.llm_calls == 3
    assert llm.requests[-1]["tools"] is None
    assert llm.requests[-1]["messages"][-1]["content"] == FINAL_ANSWER_NUDGE


def test_long_tool_output_is_truncated(retriever):
    llm = ScriptedLLM([call("search_product_catalog", query="laptop"), AssistantTurn(content="ok")])
    Agent("A", "i", [make_catalog_tool(retriever)], llm, max_tool_output_chars=50).run("task")
    assert llm.requests[1]["messages"][-1]["content"].endswith("[truncated]")


def test_team_runs_three_agents_and_passes_outputs_to_advisor(retriever, settings):
    llm = ScriptedLLM(
        [
            AssistantTurn(content="SHORTLIST: SKU-LAP-001"),
            AssistantTurn(content="CRITERIA: 16 GB RAM"),
            AssistantTurn(content="Buy SKU-LAP-001"),
        ]
    )
    web = make_web_search_tool(search_fn=lambda q: [])
    result = ShoppingResearchTeam(retriever, llm, settings, web_tool=web).run("laptop under 60000")

    assert [s.agent for s in result.steps] == ["Catalog Specialist", "Market Researcher", "Shopping Advisor"]
    assert result.answer == "Buy SKU-LAP-001"
    advisor_task = llm.requests[2]["messages"][1]["content"]
    assert "SHORTLIST: SKU-LAP-001" in advisor_task and "CRITERIA: 16 GB RAM" in advisor_task
