# File: retail-rag-agent/scripts/generate_catalog.py
"""
Generate a deterministic synthetic product catalog -> data/raw/catalog.csv

Run:  python -m scripts.generate_catalog
Same seed => same catalog every time, so evaluation numbers are reproducible.
"""

import csv
import random
from pathlib import Path

from app.config import get_settings
from scripts.catalog_spec import CATEGORY_SPECS, PRODUCTS_PER_FEATURE

SEED = 42
FIELDS = [
    "sku",
    "name",
    "category",
    "brand",
    "price_inr",
    "mrp_inr",
    "discount_percent",
    "rating",
    "review_count",
    "warranty_months",
    "in_stock",
    "specs",
    "highlight",
    "feature_id",
    "description",
]


def _price(rng: random.Random, low: int, high: int) -> int:
    """Random price that ends in 99, e.g. 54,999 (how Indian retail prices look)."""
    raw = rng.randint(low, high)
    return max(499, (raw // 1000) * 1000 + 999 if raw > 5000 else (raw // 100) * 100 + 99)


def _mrp(rng: random.Random, price: int) -> tuple[int, int]:
    """MRP (list price) above the selling price, and the resulting discount percent."""
    target = rng.choice([5, 8, 10, 12, 15, 18, 20, 25, 30])
    mrp = int(price / (1 - target / 100))
    mrp = (mrp // 100) * 100 + 99  # MRPs also end in 99
    if mrp <= price:
        mrp = price + 100
    return mrp, round((mrp - price) / mrp * 100)


def _tier(price: int, low: int, high: int) -> str:
    position = (price - low) / (high - low)
    if position < 0.25:
        return "budget-friendly"
    if position < 0.55:
        return "mid-range"
    if position < 0.8:
        return "premium"
    return "flagship"


def generate(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    products: list[dict] = []
    used_names: set[str] = set()

    for category, spec in CATEGORY_SPECS.items():
        counter = 0
        for feature_id, feature_text, _paraphrase in spec["features"]:
            for _ in range(PRODUCTS_PER_FEATURE):
                counter += 1
                brand = rng.choice(spec["brands"])
                # Near-duplicate names on purpose (e.g. "Kestrel Neo 512" vs "514").
                while True:
                    name = f"{brand} {rng.choice(spec['series'])} {rng.randint(100, 999)}"
                    if name not in used_names:
                        used_names.add(name)
                        break

                low, high = spec["price_range"]
                price = _price(rng, low, high)
                mrp, discount = _mrp(rng, price)
                specs = {key: rng.choice(values) for key, values in spec["spec_options"].items()}
                specs_text = "; ".join(f"{k}: {v}" for k, v in specs.items())
                tier = _tier(price, low, high)

                products.append(
                    {
                        "sku": f"SKU-{spec['code']}-{counter:03d}",
                        "name": name,
                        "category": category,
                        "brand": brand,
                        "price_inr": price,
                        "mrp_inr": mrp,
                        "discount_percent": discount,
                        "rating": round(rng.uniform(3.6, 4.9), 1),
                        "review_count": rng.randint(12, 4800),
                        "warranty_months": rng.choice(spec["warranty_months"]),
                        "in_stock": "yes" if rng.random() < 0.85 else "no",
                        "specs": specs_text,
                        "highlight": feature_text,
                        "feature_id": feature_id,  # used only by the eval-set builder, never shown to the LLM
                        "description": (
                            f"The {name} is a {tier} {spec['noun']} from {brand}. "
                            f"{feature_text}. It comes with {specs_text}."
                        ),
                    }
                )
    return products


def main() -> None:
    settings = get_settings()
    out_path: Path = settings.data_dir / "catalog.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    products = generate()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(products)
    print(f"Wrote {len(products)} products to {out_path}")


if __name__ == "__main__":
    main()