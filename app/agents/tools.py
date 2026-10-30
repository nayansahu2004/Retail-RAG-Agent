"""Tool definitions available to the research agents.

Each tool exposes a name, a model-facing description, a Pydantic argument
schema (used both to generate the JSON Schema sent to the LLM and to
validate incoming calls), and a Python callable. Errors are returned to the
model as text so a run can self-correct rather than crash.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from app.rag.entities import RetrievedChunk
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Bundle of metadata, argument schema, and callable that an agent can invoke."""

    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[..., str]

    def schema(self) -> dict:
        """Return the OpenAI-compatible function schema for this tool."""
        parameters = self.args_model.model_json_schema()
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": parameters},
        }

    def run(self, raw_arguments: str) -> tuple[str, bool]:
        """Validate and execute ``raw_arguments``.

        Returns:
            An ``(output, ok)`` tuple. ``ok`` is ``False`` when validation or
            execution failed; the output in that case describes the failure
            so the model can recover.
        """
        try:
            args = self.args_model.model_validate_json(raw_arguments or "{}")
        except ValidationError as exc:
            details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return f"Invalid arguments for {self.name}: {details}. Fix them and call the tool again.", False
        try:
            return self.func(**args.model_dump()), True
        except Exception as exc:
            logger.warning("tool failed", extra={"extra_fields": {"tool": self.name, "error": repr(exc)[:200]}})
            return f"Tool {self.name} failed: {type(exc).__name__}. Continue without it and say so.", False


def _format(results: list[RetrievedChunk]) -> str:
    if not results:
        return "No matching results."
    return "\n\n".join(f"[{r.chunk.id}]\n{r.chunk.text}" for r in results)


class CatalogSearchArgs(BaseModel):
    query: str = Field(description="What the customer needs in plain words, e.g. 'light laptop with long battery'")
    max_price_inr: int | None = Field(default=None, description="Optional budget ceiling in rupees, e.g. 60000")
    category: str | None = Field(default=None, description="Optional exact category name from the tool description")


def make_catalog_tool(retriever: HybridRetriever) -> Tool:
    """Build the catalog-search tool bound to ``retriever``."""
    categories = sorted({c.metadata["category"] for c in retriever.chunks if c.source_type == "product"})

    def search(query: str, max_price_inr: int | None = None, category: str | None = None) -> str:
        results, _ = retriever.retrieve(query, top_k=20, source_type="product")
        if category:
            results = [r for r in results if r.chunk.metadata.get("category", "").lower() == category.lower()]
        if max_price_inr:
            results = [r for r in results if r.chunk.metadata.get("price_inr", 0) <= max_price_inr]
        return _format(results[:5])

    return Tool(
        name="search_product_catalog",
        description=(
            "Search NovaMart's product catalog. Returns products with SKU, price, specs and highlight. "
            f"Use max_price_inr for budgets and category to narrow results (categories: {', '.join(categories)}). "
            "If results are weak, try a differently worded query."
        ),
        args_model=CatalogSearchArgs,
        func=search,
    )


class PolicySearchArgs(BaseModel):
    query: str = Field(description="The policy question, e.g. 'return window for laptops' or 'no-cost EMI banks'")


def make_policy_tool(retriever: HybridRetriever) -> Tool:
    """Build the policy-search tool bound to ``retriever``."""

    def search(query: str) -> str:
        results, _ = retriever.retrieve(query, top_k=3, source_type="policy")
        return _format(results)

    return Tool(
        name="search_store_policies",
        description=(
            "Search NovaMart's official policies: returns and refunds, shipping and delivery, warranty and "
            "repairs, payments and EMI, and Plus membership. Check a rule here before stating it."
        ),
        args_model=PolicySearchArgs,
        func=search,
    )


class WebSearchArgs(BaseModel):
    query: str = Field(description="A web search query, e.g. 'how much RAM for a machine learning laptop'")


def duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """Return DuckDuckGo results for ``query`` (region ``in-en``)."""
    from ddgs import DDGS

    return DDGS().text(query, region="in-en", safesearch="moderate", max_results=max_results)


def make_web_search_tool(search_fn: Callable[[str], list[dict]] = duckduckgo_search) -> Tool:
    """Build the web-search tool. ``search_fn`` is injectable for tests."""

    def search(query: str) -> str:
        hits = search_fn(query)
        if not hits:
            return "No web results."
        return "\n\n".join(f"{h.get('title', '')}\n{h.get('href', '')}\n{h.get('body', '')}" for h in hits)

    return Tool(
        name="web_search",
        description=(
            "Search the public web for general buying advice, which specifications matter, and typical market "
            "prices in India. Do NOT use it for NovaMart products or policies."
        ),
        args_model=WebSearchArgs,
        func=search,
    )
