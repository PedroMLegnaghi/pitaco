"""Label encoding for the BERTimbau baseline: 6 aspects × 3 sentiments = 18 independent labels.

Multi-label, not multi-class: a review can be `entrega:negativo` AND `produto:positivo` at once, so
each label gets its own sigmoid and binary cross-entropy (BCE) loss instead of one softmax.

Decoding keeps the schema rule "one sentiment per aspect": for each aspect, take the most likely
sentiment and keep it only if its probability clears the threshold. The threshold is tuned on the
silver validation split — never on gold.
"""

from collections.abc import Sequence

import numpy as np

from pitaco.schema import ASPECTS, SENTIMENTS

LABELS: tuple[str, ...] = tuple(f"{a}:{s}" for a in ASPECTS for s in SENTIMENTS)
LABEL_INDEX = {label: i for i, label in enumerate(LABELS)}


def encode(pairs: set[tuple[str, str]]) -> list[float]:
    vector = [0.0] * len(LABELS)
    for aspect, sentiment in pairs:
        vector[LABEL_INDEX[f"{aspect}:{sentiment}"]] = 1.0
    return vector


def decode(probs: Sequence[float], threshold: float) -> set[tuple[str, str]]:
    grid = np.asarray(probs, dtype=float).reshape(len(ASPECTS), len(SENTIMENTS))
    pairs = set()
    for i, aspect in enumerate(ASPECTS):
        j = int(grid[i].argmax())
        if grid[i, j] >= threshold:
            pairs.add((aspect, SENTIMENTS[j]))
    return pairs


def tune_threshold(
    probs: np.ndarray,
    gold: list[set[tuple[str, str]]],
    grid: Sequence[float] = tuple(np.round(np.arange(0.1, 0.91, 0.05), 2)),
) -> tuple[float, float]:
    """Threshold that maximizes joint macro-F1 on validation; returns (threshold, score)."""
    from pitaco.eval.metrics import joint_pairs

    ids = [str(i) for i in range(len(gold))]
    gold_by_id = dict(zip(ids, gold, strict=True))
    best = (0.5, -1.0)
    for t in grid:
        pred = {rid: decode(p, t) for rid, p in zip(ids, probs, strict=True)}
        score = joint_pairs(gold_by_id, pred, ids)["macro_f1"]
        if score > best[1]:
            best = (float(t), score)
    return best
