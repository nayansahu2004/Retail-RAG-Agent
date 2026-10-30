"""The three-agent research workflow for the shopping-advisor mode.

The workflow is fixed: a Catalog Specialist and a Market Researcher produce
independent intermediate outputs, then a Shopping Advisor consumes both and
writes the final reply. The first two can run concurrently when the
provider's rate limits allow it.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.agents.agent import Agent, AgentResult
from app.agents.tools import Tool, make_catalog_tool, make_policy_tool, make_web_search_tool
from app.config import Settings
from app.llm.client import ToolCallingLLM
from app.rag.retriever import HybridRetriever

CATALOG_INSTRUCTIONS = """You are NovaMart's Catalog Specialist.
Find the products in NovaMart's catalog that best fit the customer's needs and budget.
- Use search_product_catalog. Use max_price_inr when the customer gives a budget, and category when it is clear.
- Report ONLY products returned by the tool. Never invent products, SKUs or prices.
- Output: up to 4 products, each with name, SKU, price, key specs and one line on why it fits.
- If nothing fits, say so plainly."""

MARKET_INSTRUCTIONS = """You are a Market Researcher for consumer products in India.
Use web_search to find what matters when buying this kind of product for this use case.
- Output: 3 to 5 short buying criteria (for example a minimum RAM or battery life), each with its source URL,
  and a typical price range in India.
- Do not recommend specific stores. If web search fails, give brief general guidance and say it is unverified."""

ADVISOR_INSTRUCTIONS = """You are NovaMart's Shopping Advisor. You write the final reply to the customer.
- Recommend one best option and one alternative ONLY from the catalog shortlist you are given, with SKU and price.
- Use search_store_policies to check the policies that matter for this purchase (returns, warranty, EMI or delivery).
- Keep NovaMart facts (catalog, policies) clearly separate from general web advice.
- Never invent products, prices or policy terms. Be concise and friendly.
- Format: top pick, alternative, a 2-3 line comparison, relevant NovaMart policies, caveats."""


@dataclass
class TeamResult:
    """Combined output of a research-team run."""

    answer: str
    steps: list[AgentResult]
    total_ms: float


class ShoppingResearchTeam:
    """Three-agent orchestration: catalog + market research -> advisor."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: ToolCallingLLM,
        settings: Settings,
        web_tool: Tool | None = None,
    ):
        common = dict(
            llm=llm, max_steps=settings.agent_max_steps, max_tool_output_chars=settings.agent_tool_output_chars
        )
        self.catalog_agent = Agent("Catalog Specialist", CATALOG_INSTRUCTIONS, [make_catalog_tool(retriever)], **common)
        self.market_agent = Agent(
            "Market Researcher", MARKET_INSTRUCTIONS, [web_tool or make_web_search_tool()], **common
        )
        self.advisor_agent = Agent("Shopping Advisor", ADVISOR_INSTRUCTIONS, [make_policy_tool(retriever)], **common)
        self.parallel = settings.agent_parallel

    def run(self, question: str, history_text: str = "") -> TeamResult:
        """Run the full workflow and return the advisor's final answer."""
        started = time.perf_counter()
        request = f"Customer request: {question}\nEarlier conversation: {history_text or 'none'}"

        if self.parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                catalog_future = pool.submit(self.catalog_agent.run, request)
                market_future = pool.submit(self.market_agent.run, request)
                catalog, market = catalog_future.result(), market_future.result()
        else:
            catalog, market = self.catalog_agent.run(request), self.market_agent.run(request)

        advisor = self.advisor_agent.run(
            f"{request}\n\n<catalog_shortlist>\n{catalog.output}\n</catalog_shortlist>\n\n"
            f"<buying_criteria>\n{market.output}\n</buying_criteria>"
        )
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        return TeamResult(answer=advisor.output, steps=[catalog, market, advisor], total_ms=total_ms)
