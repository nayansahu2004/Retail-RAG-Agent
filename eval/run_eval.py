# File: retail-rag-agent/eval/run_eval.py
"""
Evaluation runner. These numbers are what you put on your resume.

  python -m eval.run_eval retrieval                  # no API keys needed, ~5 min
  python -m eval.run_eval generation --limit 40      # needs GROQ key (GEMINI key optional)

RETRIEVAL: tests many retrieval configurations on the same golden set and
picks the best one: dense only, BM25 only, weighted hybrids (BM25 weight 1.0 /
0.5 / 0.3 / 0.1, RRF k 60 / 10), with and without the BGE query instruction,
then adds the reranker to the two best. Reports Hit@1, Hit@5, Recall@5,
MRR@10, nDCG@10 and p50/p95 latency, per question type, and prints the exact
.env lines for the winner (within a latency budget, default p95 <= 300 ms).

GENERATION: runs the full pipeline (Groq) and grades the answers with an LLM
judge. If GEMINI_API_KEY is set, Gemini is the judge, so the generator does not
grade its own work (self-preference bias). Without it, Groq judges itself and
the report flags that the scores are less trustworthy.
  faithfulness   every claim is supported by the retrieved sources (no hallucination)
  correctness    the answer contains the reference fact
  citation rate  answers that cite at least one valid source
  refusal acc.   out-of-scope questions correctly refused
  false refusal  in-scope questions wrongly refused
"""

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from app.config import get_settings
from app.logging_config import setup_logging
from eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, percentile, recall_at_k

GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_golden() -> list[dict]:
    if not GOLDEN_SET.exists():
        raise SystemExit("eval/golden_set.jsonl missing. Run: python -m scripts.build_eval_set")
    with GOLDEN_SET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


# --------------------------------------------------------------- retrieval
METRICS = ("hit@1", "hit@5", "recall@5", "mrr@10", "ndcg@10")


def _candidate_configs():
    """Phase 1 grid: every cheap (no-rerank) combination worth testing."""
    from app.rag.retriever import BGE_QUERY_PREFIX, RetrievalConfig

    configs = [RetrievalConfig(dense_weight=0.0, sparse_weight=1.0)]  # BM25 only
    for prefix in ("", BGE_QUERY_PREFIX):
        configs.append(RetrievalConfig(sparse_weight=0.0, query_prefix=prefix))  # dense only
        for k in (60, 10):
            for w in (1.0, 0.5, 0.3, 0.1):
                configs.append(RetrievalConfig(sparse_weight=w, rrf_k=k, query_prefix=prefix))
    return configs


def _evaluate_config(retriever, config, answerable: list[dict]) -> dict:
    rows = []
    for item in answerable:
        started = time.perf_counter()
        results, _ = retriever.retrieve(item["question"], top_k=10, config=config)
        ms = (time.perf_counter() - started) * 1000
        ids = [r.chunk.id for r in results]
        rel = item["relevant"]
        rows.append(
            {
                "type": item["type"],
                "ms": ms,
                "hit@1": hit_at_k(ids, rel, 1),
                "hit@5": hit_at_k(ids, rel, 5),
                "recall@5": recall_at_k(ids, rel, 5),
                "mrr@10": mrr_at_k(ids, rel, 10),
                "ndcg@10": ndcg_at_k(ids, rel, 10),
            }
        )
    return {"label": config.label, "config": asdict(config), **_aggregate(rows)}


def _rank_key(result: dict) -> tuple:
    """Best = highest Hit@5 (is the answer in what the LLM sees?), then MRR, then lowest p95 latency."""
    o = result["overall"]
    return (o["hit@5"], o["mrr@10"], o["ndcg@10"], -result["latency_ms"]["p95"])


def run_retrieval(max_p95_ms: float) -> None:
    from app.rag.retriever import build_retriever

    settings = get_settings()
    retriever = build_retriever(settings, with_reranker=True)
    items = load_golden()
    answerable = [i for i in items if i["type"] != "out_of_scope"]
    retriever.retrieve("warm up the models")  # first call loads ONNX sessions; exclude from latency

    print(f"Phase 1: testing {len(_candidate_configs())} fast configurations on {len(answerable)} questions...")
    results = [_evaluate_config(retriever, cfg, answerable) for cfg in _candidate_configs()]

    # Phase 2: reranking is slow, so only try it on top of the 2 best fast configurations.
    top_two = sorted(results, key=_rank_key, reverse=True)[:2]
    print("Phase 2: testing the reranker on the two best configurations (slower)...")
    for result in top_two:
        cfg = replace(_config_from_dict(result["config"]), rerank=True)
        results.append(_evaluate_config(retriever, cfg, answerable))

    results.sort(key=_rank_key, reverse=True)
    within_budget = [r for r in results if r["latency_ms"]["p95"] <= max_p95_ms]
    best = within_budget[0] if within_budget else results[0]

    report = {
        "n_questions": len(answerable),
        "max_p95_ms": max_p95_ms,
        "best": best,
        "results": results,
    }
    if best["config"]["rerank"]:
        report["rerank_calibration"] = _rerank_calibration(retriever, items, _config_from_dict(best["config"]))
    _print_retrieval(report)
    print(f"\nSaved: {_save('retrieval', report)}")


def _config_from_dict(d: dict):
    from app.rag.retriever import RetrievalConfig

    return RetrievalConfig(**d)


def _aggregate(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[row["type"]].append(row)
    out = {g: {m: _mean([r[m] for r in rs]) for m in METRICS} for g, rs in groups.items()}
    latencies = [r["ms"] for r in rows]
    out["latency_ms"] = {"p50": round(percentile(latencies, 50), 1), "p95": round(percentile(latencies, 95), 1)}
    return out


def _rerank_calibration(retriever, items: list[dict], config) -> dict:
    """Top-1 rerank score for answerable vs out-of-scope questions. A good threshold separates them."""
    scores: dict[str, list[float]] = {"answerable": [], "out_of_scope": []}
    for item in items:
        results, _ = retriever.retrieve(item["question"], top_k=1, config=config)
        if results and results[0].rerank_score is not None:
            key = "out_of_scope" if item["type"] == "out_of_scope" else "answerable"
            scores[key].append(results[0].rerank_score)
    return {
        key: {"p5": round(percentile(v, 5), 2), "p50": round(percentile(v, 50), 2), "p95": round(percentile(v, 95), 2)}
        for key, v in scores.items()
    }


def _print_retrieval(report: dict) -> None:
    results = report["results"]
    print(f"\nRetrieval evaluation on {report['n_questions']} answerable questions (sorted best first)\n")
    print("| config | Hit@1 | Hit@5 | Recall@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        o, lat = r["overall"], r["latency_ms"]
        print(
            f"| {r['label']} | {o['hit@1']} | {o['hit@5']} | {o['recall@5']} | {o['mrr@10']} | {o['ndcg@10']} "
            f"| {lat['p50']} | {lat['p95']} |"
        )

    types = [t for t in ("exact", "sku", "semantic", "policy") if t in results[0]]
    shown = results[:6]
    baselines = [r for r in results if r["label"] in ("dense", "bm25") and r not in shown]
    print("\nHit@5 by question type (top 6 + baselines):\n")
    print("| config | " + " | ".join(types) + " |")
    print("|---|" + "---|" * len(types))
    for r in shown + baselines:
        print(f"| {r['label']} | " + " | ".join(str(r[t]["hit@5"]) for t in types) + " |")

    best = report["best"]
    cfg = best["config"]
    print(f"\nBEST within p95 <= {report['max_p95_ms']:g} ms: {best['label']}")
    print(f"  Hit@5 {best['overall']['hit@5']}, MRR@10 {best['overall']['mrr@10']}, p95 {best['latency_ms']['p95']} ms")
    print("\nPut these lines in your .env to use it:\n")
    print(f"RETRIEVAL_DENSE_WEIGHT={cfg['dense_weight']:g}")
    print(f"RETRIEVAL_SPARSE_WEIGHT={cfg['sparse_weight']:g}")
    print(f"RRF_K={cfg['rrf_k']}")
    print(f"RETRIEVAL_RERANK={str(cfg['rerank']).lower()}")
    print(f'EMBEDDING_QUERY_PREFIX="{cfg["query_prefix"]}"')
    if "rerank_calibration" in report:
        cal = report["rerank_calibration"]
        print(f"\nTop-1 rerank score  answerable: {cal['answerable']}  out-of-scope: {cal['out_of_scope']}")
        print("Set MIN_RERANK_SCORE between out-of-scope p95 and answerable p5 only if they do not overlap.")


# --------------------------------------------------------------- generation
JUDGE_PROMPT = """You are a strict evaluator for a retail shopping chatbot.
Return only JSON: {"faithful": true|false, "correct": true|false, "reason": "<one short sentence>"}
- faithful: every factual claim in ANSWER is supported by SOURCES. Ignore citation formatting.
- correct: ANSWER contains the key facts of REFERENCE and does not contradict it.
  If REFERENCE starts with "Any of:", correct means ANSWER recommends at least one listed product."""


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"faithful": False, "correct": False, "reason": f"unparseable judge output: {text[:100]}"}


def _stratified(items: list[dict], limit: int) -> list[dict]:
    """Round-robin across question types so a small sample still covers every type."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        buckets[item["type"]].append(item)
    picked: list[dict] = []
    while len(picked) < limit and any(buckets.values()):
        for bucket in buckets.values():
            if bucket and len(picked) < limit:
                picked.append(bucket.pop(0))
    return picked


def run_generation(limit: int, sleep_s: float) -> None:
    from app.llm.client import LLMError, build_llm
    from app.memory import SessionStore
    from app.rag.pipeline import RAGPipeline
    from app.rag.prompts import format_sources
    from app.rag.retriever import build_retriever

    settings = get_settings()
    if not settings.groq_api_key:
        raise SystemExit("Generation eval needs GROQ_API_KEY.")
    independent_judge = bool(settings.gemini_api_key)
    if not independent_judge:
        print("WARNING: no GEMINI_API_KEY, so Groq grades its own answers. Treat scores as optimistic.\n")

    # Generator uses Groq only, so every scored answer comes from the production model.
    pipeline = RAGPipeline(build_retriever(settings), build_llm(settings, ("groq",)), SessionStore(), settings)
    judge = build_llm(settings, order=("gemini",) if independent_judge else ("groq",))

    records = []
    for n, item in enumerate(_stratified(load_golden(), limit), start=1):
        try:
            result = pipeline.answer(item["question"], session_id=f"eval-{item['id']}")
        except LLMError as exc:  # e.g. rate limit on both providers: skip, do not crash the run
            print(f"[{n}/{limit}] skipped ({exc})")
            continue
        record = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "answer": result.answer,
            "coverage": result.coverage,
            "refused": result.refused,
            "cited": bool(result.cited),
            "llm": result.llm,
            "latency_ms": result.timings_ms.get("total_ms", 0.0),
        }
        if item["type"] != "out_of_scope" and not result.refused:
            verdict = judge.generate(
                JUDGE_PROMPT,
                [
                    {
                        "role": "user",
                        "content": f"SOURCES:\n{format_sources(result.results)}\n\nQUESTION: {item['question']}"
                        f"\n\nANSWER: {result.answer}\n\nREFERENCE: {item['reference']}",
                    }
                ],
                temperature=0.0,
                json_mode=True,
            )
            record.update(_parse_json(verdict.text))
        records.append(record)
        print(
            f"[{n}/{limit}] {item['type']:<12} refused={result.refused!s:<5} "
            f"faithful={record.get('faithful', '-')!s:<5} correct={record.get('correct', '-')}"
        )
        time.sleep(sleep_s)  # Groq free tier: 30 requests and 8K tokens per minute

    answerable = [r for r in records if r["type"] != "out_of_scope"]
    answered = [r for r in answerable if not r["refused"]]
    oos = [r for r in records if r["type"] == "out_of_scope"]
    latencies = [r["latency_ms"] for r in records]
    summary = {
        "n": len(records),
        "judge": judge.providers[0].name + ("" if independent_judge else " (self-graded)"),
        "faithfulness": _mean([float(r.get("faithful", False)) for r in answered]),
        "answer_correctness": _mean([float(r.get("correct", False)) for r in answered]),
        "citation_rate": _mean([float(r["cited"]) for r in answered]),
        "false_refusal_rate": _mean([float(r["refused"]) for r in answerable]),
        "partial_answer_rate": _mean([float(r["coverage"] == "partial") for r in answerable]),
        "out_of_scope_refusal_accuracy": _mean([float(r["refused"]) for r in oos]),
        "latency_ms": {"p50": round(percentile(latencies, 50)), "p95": round(percentile(latencies, 95))},
        "llm_usage": {k: sum(r["llm"] == k for r in records) for k in {r["llm"] for r in records}},
    }
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved: {_save('generation', {'summary': summary, 'records': records})}")


def main() -> None:
    setup_logging("WARNING")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ret = sub.add_parser("retrieval", help="Find the best retrieval configuration (no API keys needed)")
    ret.add_argument("--max-p95-ms", type=float, default=300.0, help="Latency budget for picking the best config")
    gen = sub.add_parser("generation", help="End-to-end answer quality with an LLM judge")
    gen.add_argument("--limit", type=int, default=40)
    gen.add_argument("--sleep", type=float, default=12.0, help="Seconds between questions (free-tier rate limits)")
    args = parser.parse_args()
    if args.command == "retrieval":
        run_retrieval(args.max_p95_ms)
    else:
        run_generation(args.limit, args.sleep)


if __name__ == "__main__":
    main()