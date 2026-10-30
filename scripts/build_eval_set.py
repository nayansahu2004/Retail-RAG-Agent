# File: retail-rag-agent/scripts/build_eval_set.py
"""
Build the "golden" evaluation set -> eval/golden_set.jsonl

Five question types, each testing a different weakness:
  exact        "What is the price of the Kestrel Neo 514?"   -> product named in full
  sku          "Is SKU-LAP-017 in stock?"                     -> bare product code (BM25 strength)
  semantic     "a laptop that makes zero noise under load"   -> needs meaning (embedding strength)
  policy       customer-style policy questions                -> needs the right section
  out_of_scope questions the bot must refuse                  -> tests hallucination guardrails

Run:  python -m scripts.build_eval_set
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from app.config import get_settings
from eval.policy_questions import OUT_OF_SCOPE_QUESTIONS, POLICY_QUESTIONS
from scripts.catalog_spec import CATEGORY_SPECS

SEED = 7
N_EXACT = 40
N_SKU = 30
FEATURES_PER_CATEGORY = 5
OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "golden_set.jsonl"

EXACT_TEMPLATES = [
    ("What is the price of the {name}?", lambda p: f"₹{int(p['price_inr']):,}"),
    ("How long is the warranty on the {name}?", lambda p: f"{p['warranty_months']} months"),
    ("What rating do customers give the {name}?", lambda p: f"{p['rating']} out of 5"),
    ("Is the {name} in stock right now?", lambda p: "Yes" if p["in_stock"] == "yes" else "No"),
    ("What discount is there on the {name}?", lambda p: f"{p['discount_percent']}% off MRP ₹{int(p['mrp_inr']):,}"),
]


def main() -> None:
    rng = random.Random(SEED)
    catalog_path = get_settings().data_dir / "catalog.csv"
    with catalog_path.open(encoding="utf-8") as f:
        products = list(csv.DictReader(f))

    items: list[dict] = []

    for i, product in enumerate(rng.sample(products, N_EXACT)):
        template, reference = EXACT_TEMPLATES[i % len(EXACT_TEMPLATES)]
        items.append(
            {
                "type": "exact",
                "question": template.format(name=product["name"]),
                "relevant": [f"product::{product['sku']}"],
                "reference": reference(product),
            }
        )

    SKU_TEMPLATES = [
        ("What is the price of {sku}?", lambda p: f"₹{int(p['price_inr']):,}"),
        ("Is {sku} in stock?", lambda p: "Yes" if p["in_stock"] == "yes" else "No"),
        ("Which brand makes {sku}?", lambda p: p["brand"]),
    ]
    for i, product in enumerate(rng.sample(products, N_SKU)):
        template, reference = SKU_TEMPLATES[i % len(SKU_TEMPLATES)]
        items.append(
            {
                "type": "sku",
                "question": template.format(sku=product["sku"]),
                "relevant": [f"product::{product['sku']}"],
                "reference": reference(product),
            }
        )

    by_feature: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for product in products:
        by_feature[(product["category"], product["feature_id"])].append(product)

    for category, spec in CATEGORY_SPECS.items():
        for feature_id, _text, paraphrase in rng.sample(spec["features"], FEATURES_PER_CATEGORY):
            matches = by_feature[(category, feature_id)]
            items.append(
                {
                    "type": "semantic",
                    "question": f"I'm looking for a new {spec['noun']}. Must have: {paraphrase}. What do you recommend?",
                    "relevant": [f"product::{p['sku']}" for p in matches],
                    "reference": "Any of: " + ", ".join(f"{p['name']} ({p['sku']})" for p in matches),
                }
            )

    items.extend({"type": "policy", **q} for q in POLICY_QUESTIONS)
    items.extend(
        {"type": "out_of_scope", "question": q, "relevant": [], "reference": "REFUSE"} for q in OUT_OF_SCOPE_QUESTIONS
    )

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            f.write(json.dumps({"id": f"q{idx:03d}", **item}, ensure_ascii=False) + "\n")

    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item["type"]] += 1
    print(f"Wrote {len(items)} questions to {OUT_PATH}: {dict(counts)}")


if __name__ == "__main__":
    main()