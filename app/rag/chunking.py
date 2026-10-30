# File: retail-rag-agent/app/rag/chunking.py
"""
Turning raw data into chunks.

Two strategies, because the data has two shapes:
1. Products (structured rows) -> ONE chunk per product. A product is already a
   small, self-contained unit; splitting it would separate its name from its price.
2. Policies (long markdown) -> split by "## " section, then by size with overlap.
   Each chunk is prefixed with "Document > Section" (a "contextual header") so a
   chunk like "It costs ₹99" still says WHAT costs ₹99. This measurably helps
   both keyword and vector search.
"""

import csv
import re
from pathlib import Path

from app.rag.entities import Chunk

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ---------------------------------------------------------------- products
def product_to_chunk(row: dict) -> Chunk:
    price = int(row["price_inr"])
    mrp = int(row.get("mrp_inr") or price)
    price_line = f"Price: ₹{price:,}"
    if mrp > price:
        price_line += f" (MRP ₹{mrp:,}, {row.get('discount_percent', round((mrp - price) / mrp * 100))}% off)"
    text = (
        f"Product: {row['name']}\n"
        f"SKU: {row['sku']}\n"
        f"Category: {row['category']}\n"
        f"Brand: {row['brand']}\n"
        f"{price_line}\n"
        f"Rating: {row['rating']}/5 from {row['review_count']} reviews\n"
        f"Warranty: {row['warranty_months']} months\n"
        f"In stock: {row['in_stock']}\n"
        f"Specs: {row['specs']}\n"
        f"Highlight: {row['highlight']}\n"
        f"Description: {row['description']}"
    )
    metadata = {
        "source_type": "product",
        "title": row["name"],
        "sku": row["sku"],
        "category": row["category"],
        "brand": row["brand"],
        "price_inr": price,
        "mrp_inr": mrp,
        "in_stock": row["in_stock"] == "yes",
    }
    return Chunk(id=f"product::{row['sku']}", text=text, metadata=metadata)


def load_catalog(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [product_to_chunk(row) for row in csv.DictReader(f)]


# ---------------------------------------------------------------- policies
def split_markdown_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (document title, [(section title, section body), ...])."""
    markdown = _HTML_COMMENT.sub("", markdown).strip()
    doc_title = "Untitled"
    sections: list[tuple[str, list[str]]] = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
    return doc_title, [(title, "\n".join(body).strip()) for title, body in sections if "\n".join(body).strip()]


def split_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    """
    Paragraph-aware splitter with overlap.
    Packs whole paragraphs (or sentences, if a paragraph is too long) into chunks
    of at most max_chars. The last `overlap` chars of a chunk are repeated at the
    start of the next one so a fact cut at the boundary is not lost.
    """
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        units.extend([para] if len(para) <= max_chars else [s for s in _SENTENCE_END.split(para) if s])

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if " " in tail:  # start the overlap on a word boundary
                tail = tail[tail.index(" ") + 1 :]
            current = f"{tail} {unit}".strip()
        else:
            current = unit
        while len(current) > max_chars:  # a single sentence longer than max_chars
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap :]
    if current:
        chunks.append(current)
    return chunks


def policy_to_chunks(path: Path, max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    doc_title, sections = split_markdown_sections(path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for section_title, body in sections:
        for i, part in enumerate(split_text(body, max_chars, overlap)):
            chunks.append(
                Chunk(
                    id=f"policy::{path.stem}::{slugify(section_title)}::{i}",
                    text=f"{doc_title} > {section_title}\n{part}",
                    metadata={
                        "source_type": "policy",
                        "title": f"{doc_title}: {section_title}",
                        "doc": path.stem,
                        "section": slugify(section_title),
                    },
                )
            )
    return chunks


def load_corpus(data_dir: Path, max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    catalog = data_dir / "catalog.csv"
    if not catalog.exists():
        raise FileNotFoundError(f"{catalog} not found. Run: python -m scripts.generate_catalog")
    chunks = load_catalog(catalog)
    for md in sorted((data_dir / "policies").glob("*.md")):
        chunks.extend(policy_to_chunks(md, max_chars, overlap))
    return chunks