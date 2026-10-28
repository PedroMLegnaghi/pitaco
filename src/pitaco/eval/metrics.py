"""Metrics for aspect-based sentiment, computed per review over (aspect, sentiment) pairs.

Inputs are `gold[id]` and `pred[id]`: sets of (aspect, sentiment) pairs. A missing or invalid
prediction is an empty set, so it costs recall instead of disappearing from the denominator.
`ids` is a list (not a set): the bootstrap resamples reviews with replacement and duplicates must
count twice.

- Aspect detection: is the aspect present at all? per-aspect P/R/F1, macro and micro F1.
- Joint (aspect, sentiment): a hit needs both right. A wrong sentiment is one FP and one FN.
- Sentiment on matched aspects: polarity accuracy/macro-F1 where both sides found the aspect —
  isolates "reads the polarity right" from "finds the aspect".
- Cohen's kappa per aspect (absent/positivo/neutro/negativo): agreement corrected for chance,
  used for teacher–human and human–human agreement.

Macro F1 skips an aspect only when it has no gold support and no predictions (undefined).
"""

from collections import Counter
from collections.abc import Sequence

from pitaco.schema import ASPECTS, SENTIMENTS

Pairs = set[tuple[str, str]]
ABSENT = "absent"


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _by_aspect(pairs: Pairs) -> dict[str, str]:
    return {aspect: sentiment for aspect, sentiment in pairs}


def _macro(per_aspect: dict[str, dict]) -> float:
    defined = [m["f1"] for m in per_aspect.values() if m["support"] or m["predicted"]]
    return sum(defined) / len(defined) if defined else 0.0


def aspect_detection(gold: dict[str, Pairs], pred: dict[str, Pairs], ids: Sequence[str]) -> dict:
    counts = {a: Counter() for a in ASPECTS}
    for rid in ids:
        g, p = set(_by_aspect(gold[rid])), set(_by_aspect(pred.get(rid, set())))
        for a in ASPECTS:
            counts[a]["tp"] += a in g and a in p
            counts[a]["fp"] += a in p and a not in g
            counts[a]["fn"] += a in g and a not in p
    per_aspect = {}
    for a, c in counts.items():
        p_, r_, f_ = prf(c["tp"], c["fp"], c["fn"])
        per_aspect[a] = {
            "precision": p_,
            "recall": r_,
            "f1": f_,
            "support": c["tp"] + c["fn"],
            "predicted": c["tp"] + c["fp"],
        }
    total = sum(counts.values(), Counter())
    return {
        "per_aspect": per_aspect,
        "macro_f1": _macro(per_aspect),
        "micro_f1": prf(total["tp"], total["fp"], total["fn"])[2],
    }


def joint_pairs(gold: dict[str, Pairs], pred: dict[str, Pairs], ids: Sequence[str]) -> dict:
    counts = {a: Counter() for a in ASPECTS}
    for rid in ids:
        g, p = _by_aspect(gold[rid]), _by_aspect(pred.get(rid, set()))
        for a in ASPECTS:
            hit = a in g and a in p and g[a] == p[a]
            counts[a]["tp"] += hit
            counts[a]["fp"] += a in p and not hit
            counts[a]["fn"] += a in g and not hit
    per_aspect = {}
    for a, c in counts.items():
        p_, r_, f_ = prf(c["tp"], c["fp"], c["fn"])
        per_aspect[a] = {
            "precision": p_,
            "recall": r_,
            "f1": f_,
            "support": c["tp"] + c["fn"],
            "predicted": c["tp"] + c["fp"],
        }
    total = sum(counts.values(), Counter())
    return {
        "per_aspect": per_aspect,
        "macro_f1": _macro(per_aspect),
        "micro_f1": prf(total["tp"], total["fp"], total["fn"])[2],
    }


def sentiment_on_matched(
    gold: dict[str, Pairs], pred: dict[str, Pairs], ids: Sequence[str]
) -> dict:
    y_true, y_pred = [], []
    for rid in ids:
        g, p = _by_aspect(gold[rid]), _by_aspect(pred.get(rid, set()))
        for a in g.keys() & p.keys():
            y_true.append(g[a])
            y_pred.append(p[a])
    if not y_true:
        return {"n": 0, "accuracy": 0.0, "macro_f1": 0.0}
    f1s = []
    for s in SENTIMENTS:
        if s in y_true or s in y_pred:
            tp = sum(t == s and q == s for t, q in zip(y_true, y_pred, strict=True))
            fp = sum(t != s and q == s for t, q in zip(y_true, y_pred, strict=True))
            fn = sum(t == s and q != s for t, q in zip(y_true, y_pred, strict=True))
            f1s.append(prf(tp, fp, fn)[2])
    accuracy = sum(t == q for t, q in zip(y_true, y_pred, strict=True)) / len(y_true)
    return {"n": len(y_true), "accuracy": accuracy, "macro_f1": sum(f1s) / len(f1s)}


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def kappa_per_aspect(gold: dict[str, Pairs], pred: dict[str, Pairs], ids: Sequence[str]) -> dict:
    per_aspect = {}
    for a in ASPECTS:
        g = [_by_aspect(gold[rid]).get(a, ABSENT) for rid in ids]
        p = [_by_aspect(pred.get(rid, set())).get(a, ABSENT) for rid in ids]
        per_aspect[a] = cohen_kappa(g, p)
    return {"per_aspect": per_aspect, "mean": sum(per_aspect.values()) / len(per_aspect)}


def compute_all(gold: dict[str, Pairs], pred: dict[str, Pairs], ids: Sequence[str]) -> dict:
    detection = aspect_detection(gold, pred, ids)
    joint = joint_pairs(gold, pred, ids)
    sentiment = sentiment_on_matched(gold, pred, ids)
    kappa = kappa_per_aspect(gold, pred, ids)
    return {
        "aspect_macro_f1": detection["macro_f1"],
        "aspect_micro_f1": detection["micro_f1"],
        "joint_macro_f1": joint["macro_f1"],
        "joint_micro_f1": joint["micro_f1"],
        "sentiment_accuracy_matched": sentiment["accuracy"],
        "sentiment_macro_f1_matched": sentiment["macro_f1"],
        "kappa_mean": kappa["mean"],
        "detail": {"detection": detection, "joint": joint, "sentiment": sentiment, "kappa": kappa},
    }


HEADLINE = ("aspect_macro_f1", "joint_macro_f1", "joint_micro_f1", "sentiment_macro_f1_matched")
