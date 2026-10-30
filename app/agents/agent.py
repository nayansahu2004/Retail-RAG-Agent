"""A minimal tool-calling agent loop.

Each ``Agent`` keeps a running message list, invokes the LLM with its tool
schemas, executes any requested tool calls, and feeds the results back until
the model answers in plain text or the step budget is exhausted. Every tool
invocation is recorded on an ``AgentResult`` for logging and UI display.
"""

import logging
import time
from dataclasses import dataclass, field

from app.agents.tools import Tool
from app.llm.client import ToolCallingLLM

logger = logging.getLogger(__name__)

FINAL_ANSWER_NUDGE = "You have used all your tool calls. Write your final answer now, using only what you found."


@dataclass
class ToolTrace:
    """Record of a single tool invocation."""

    tool: str
    arguments: str
    ok: bool
    ms: float


@dataclass
class AgentResult:
    """Terminal state of an agent run."""

    agent: str
    output: str
    tool_calls: list[ToolTrace] = field(default_factory=list)
    llm_calls: int = 0
    ms: float = 0.0


class Agent:
    """A single tool-using agent driven by a tool-calling LLM."""

    def __init__(
        self,
        name: str,
        instructions: str,
        tools: list[Tool],
        llm: ToolCallingLLM,
        max_steps: int = 4,
        max_tool_output_chars: int = 2500,
    ):
        self.name = name
        self.instructions = instructions
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm
        self.max_steps = max_steps
        self.max_tool_output_chars = max_tool_output_chars

    def run(self, task: str) -> AgentResult:
        """Execute the agent against ``task`` and return its final result."""
        started = time.perf_counter()
        messages: list[dict] = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": task},
        ]
        schemas = [tool.schema() for tool in self.tools.values()] or None
        traces: list[ToolTrace] = []
        llm_calls = 0

        for _ in range(self.max_steps):
            turn = self.llm.chat(messages, tools=schemas)
            llm_calls += 1
            if not turn.tool_calls:
                return self._result(turn.content, traces, llm_calls, started)

            messages.append(
                {
                    "role": "assistant",
                    "content": turn.content or "",
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                        for c in turn.tool_calls
                    ],
                }
            )
            for call in turn.tool_calls:
                output, ok, ms = self._execute(call.name, call.arguments)
                traces.append(ToolTrace(tool=call.name, arguments=call.arguments, ok=ok, ms=ms))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

        messages.append({"role": "user", "content": FINAL_ANSWER_NUDGE})
        turn = self.llm.chat(messages, tools=None)
        return self._result(turn.content, traces, llm_calls + 1, started)

    def _execute(self, name: str, arguments: str) -> tuple[str, bool, float]:
        started = time.perf_counter()
        tool = self.tools.get(name)
        if tool is None:
            output, ok = f"Unknown tool '{name}'. Available tools: {', '.join(self.tools)}.", False
        else:
            output, ok = tool.run(arguments)
        if len(output) > self.max_tool_output_chars:
            output = output[: self.max_tool_output_chars] + "\n[truncated]"
        return output, ok, round((time.perf_counter() - started) * 1000, 2)

    def _result(self, output: str, traces: list[ToolTrace], llm_calls: int, started: float) -> AgentResult:
        ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "agent finished",
            extra={"extra_fields": {"agent": self.name, "tool_calls": len(traces), "llm_calls": llm_calls, "ms": ms}},
        )
        return AgentResult(agent=self.name, output=output.strip(), tool_calls=traces, llm_calls=llm_calls, ms=ms)
