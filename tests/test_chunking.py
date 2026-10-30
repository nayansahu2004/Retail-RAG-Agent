import pytest

from app.rag.chunking import policy_to_chunks, product_to_chunk, split_markdown_sections, split_text


def test_short_text_is_one_chunk():
    assert split_text("hello world", max_chars=50, overlap=10) == ["hello world"]


def test_long_text_respects_max_and_overlaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = split_text(text, max_chars=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[1].split()[0] in chunks[0]


def test_overlap_must_be_smaller_than_max():
    with pytest.raises(ValueError):
        split_text("x" * 100, max_chars=10, overlap=10)


def test_markdown_sections_and_comment_removed():
    md = "<!-- File: x -->\n# Doc\n## First part\nAlpha.\n## Second\nBeta."
    title, sections = split_markdown_sections(md)
    assert title == "Doc"
    assert sections == [("First part", "Alpha."), ("Second", "Beta.")]


def test_policy_chunks_have_contextual_header(tmp_path):
    path = tmp_path / "returns.md"
    path.write_text("# Returns Policy\n## Electronics returns\nSeven days.", encoding="utf-8")
    [chunk] = policy_to_chunks(path)
    assert chunk.id == "policy::returns::electronics-returns::0"
    assert chunk.text.startswith("Returns Policy > Electronics returns")


def test_product_chunk_formats_price():
    row = {
        "sku": "SKU-1",
        "name": "Test Phone",
        "category": "Smartphones",
        "brand": "B",
        "price_inr": "54999",
        "rating": "4.5",
        "review_count": "10",
        "warranty_months": "12",
        "in_stock": "yes",
        "specs": "RAM: 8 GB",
        "highlight": "Big battery",
        "description": "Nice.",
    }
    chunk = product_to_chunk(row)
    assert "₹54,999" in chunk.text and chunk.metadata["price_inr"] == 54999
