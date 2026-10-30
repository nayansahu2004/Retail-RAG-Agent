"""Sparse keyword index backed by Okapi BM25."""

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "what",
        "which",
        "who",
        "how",
        "does",
        "do",
        "i",
        "my",
        "me",
        "you",
        "your",
        "it",
        "its",
        "this",
        "that",
        "from",
        "at",
        "by",
        "as",
        "can",
        "any",
        "have",
        "has",
        "will",
        "would",
        "should",
    ]
)
# Keeps SKUs and decimals together: "sku-lap-007", "2.5", "5g".
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip stopwords, and split on alphanumeric spans."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Store:
    """BM25 keyword index persisted as the raw tokenised corpus."""

    def __init__(self, tokenized_corpus: list[list[str]]):
        self._tokens = tokenized_corpus
        self._bm25 = BM25Okapi(tokenized_corpus)

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Store":
        """Tokenise ``texts`` and build the index."""
        return cls([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return the top ``k`` ``(row_id, score)`` matches for ``query``."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self._tokens), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Store":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return len(self._tokens)
