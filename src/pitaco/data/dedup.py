"""Duplicate detection.

Two levels:
- exact duplicates after normalization ("Ótimo!!" == "otimo") are dropped from the whole corpus;
- near-duplicates of gold reviews (fuzzy ratio >= threshold) are dropped from the silver pool, so
  the model never trains on a paraphrase of a test review.
"""

import re
import unicodedata

import numpy as np
from rapidfuzz import fuzz, process

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def dedup_key(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse spaces."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_ALNUM.sub(" ", no_accents).strip()


def near_duplicate_mask(queries: list[str], pool: list[str], threshold: int) -> np.ndarray:
    """Boolean mask over `pool`: True where an item is >= `threshold` similar to any query."""
    if not queries or not pool:
        return np.zeros(len(pool), dtype=bool)
    scores = process.cdist(
        queries, pool, scorer=fuzz.ratio, score_cutoff=threshold, dtype=np.uint8, workers=-1
    )
    return (scores >= threshold).any(axis=0)
