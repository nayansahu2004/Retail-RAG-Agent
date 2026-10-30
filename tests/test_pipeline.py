# File: retail-rag-agent/tests/test_pipeline.py
import json

from app.llm.client import LLMError
from app.rag.pipeline import extract_citations, merge_round_robin, parse_answer, parse_queries


def answer_json(text: str, coverage: str = "full") -> str:
    return json.dumps({"answer": text, "coverage": coverage})


# ------------------------------------------------------------------ pure helpers
def test_extract_citations_ignores_invalid_numbers():
    assert extract_citations("A [1], B [3], C [9]", n_sources=3) == {1, 3}


def test_parse_answer_reads_json_and_tolerates_plain_text():
    assert parse_answer(answer_json("Yes [1].", "partial")) == ("Yes [1].", "partial")
    assert parse_answer('```json\n{"answer": "Ok", "coverage": "none"}\n```') == ("Ok", "none")
    assert parse_answer("Plain text reply") == ("Plain text reply", "full")
    assert parse_answer(json.dumps({"answer": "Hi", "coverage": "weird"})) == ("Hi", "full")


def test_parse_queries_validates_dedupes_and_caps():
    raw = json.dumps({"queries": ["a laptop", "a laptop", "x", "free delivery", "emi", "extra"]})
    assert parse_queries(raw, "fallback", max_queries=3) == ["a laptop", "free delivery", "emi"]
    assert parse_queries("not json", "fallback", 3) == ["fallback"]
    assert parse_queries(json.dumps({"queries": []}), "fallback", 3) == ["fallback"]


def test_merge_round_robin_covers_every_query(retriever):
    product, _ = retriever.retrieve("Kestrel Neo 514", top_k=3)
    policy, _ = retriever.retrieve("free delivery", top_k=3, source_type="policy")
    merged = merge_round_robin([product, policy], limit=2)
    assert [m.chunk.id for m in merged] == [product[0].chunk.id, policy[0].chunk.id]


# ------------------------------------------------------------------ pipeline
def test_answer_plans_retrieves_and_cites(pipeline, fake_llm):
    result = pipeline.answer("What is the return window?", "s1")
    assert result.cited == {1} and result.coverage == "full" and not result.refused
    assert result.search_queries == ["What is the return window?"]
    assert fake_llm.kinds() == ["plan", "answer"]
    assert all(c["json_mode"] for c in fake_llm.calls)
    assert "plan_ms" in result.timings_ms


def test_multi_part_question_gets_evidence_for_each_part(pipeline, fake_llm):
    fake_llm.queries = ["Kestrel Neo 514 features", "free delivery minimum order"]
    result = pipeline.answer("Features of Kestrel Neo 514 and is delivery free?", "s2")
    ids = [r.chunk.id for r in result.results]
    assert "product::SKU-LAP-001" in ids and "policy::shipping::shipping-charges::0" in ids
    assert result.search_queries == fake_llm.queries


def test_partial_coverage_is_reported(pipeline, fake_llm):
    fake_llm.reply = answer_json("It has 8 GB RAM [1]. Discount details are not available.", "partial")
    result = pipeline.answer("RAM and discount on Kestrel Neo 514?", "s3")
    assert result.coverage == "partial" and not result.refused


def test_planning_off_skips_planner_on_first_turn_but_not_on_follow_ups(pipeline, fake_llm, settings):
    settings.query_planning = False
    pipeline.answer("Tell me about the Kestrel Neo 514", "s4")
    assert fake_llm.kinds() == ["answer"]
    pipeline.answer("Is it silent?", "s4")  # follow-up: references must still be resolved
    assert fake_llm.kinds() == ["answer", "plan", "answer"]


def test_planner_failure_falls_back_to_raw_question(pipeline, fake_llm):
    fake_llm.plan_reply = "sorry, no JSON"
    assert pipeline.answer("free delivery?", "s5").search_queries == ["free delivery?"]

    class Broken:
        def generate(self, *args, **kwargs):
            raise LLMError("down")

    pipeline.llm = Broken()
    assert pipeline.plan("is it silent?", []) == ["is it silent?"]


def test_low_rerank_score_answers_not_found_without_answer_call(pipeline, fake_llm, settings):
    settings.min_rerank_score = 100.0  # impossible to reach
    result = pipeline.answer("anything", "s6")
    assert result.refused and "answer" not in fake_llm.kinds()
    assert settings.support_contact in result.answer


def test_category_list_is_always_in_the_answer_prompt(pipeline, fake_llm):
    pipeline.answer("Do you sell refrigerators?", "s7")
    answer_call = next(c for c in fake_llm.calls if c["kind"] == "answer")
    assert "Laptops, Running Shoes" in answer_call["system"]
    assert "does not sell it" in answer_call["system"]