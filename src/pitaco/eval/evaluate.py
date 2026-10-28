"""`pitaco evaluate` and `pitaco report`: score any system's predictions against gold labels.

Every evaluation on gold test is appended to `results/test_access.log`. The log makes test-set use
auditable: if it shows dozens of test evaluations of one system, the test set was used for tuning.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pitaco.eval.bootstrap import bootstrap_ci, paired_bootstrap
from pitaco.eval.metrics import HEADLINE, Pairs, compute_all
from pitaco.labeling.annotations import load_annotations
from pitaco.predictions import PredictionRecord, read_predictions

GOLD_SPLITS = {"gold_dev": "gold_dev.jsonl", "gold_test": "gold_test.jsonl"}


def load_gold(annotations_dir: Path, split: str) -> dict[str, Pairs]:
    if split not in GOLD_SPLITS:
        raise ValueError(f"gold split must be one of {sorted(GOLD_SPLITS)}")
    annotations = load_annotations(annotations_dir / GOLD_SPLITS[split])
    if not annotations:
        raise FileNotFoundError(f"no gold annotations for {split} in {annotations_dir}")
    return {rid: ann.pairs() for rid, ann in annotations.items()}


def efficiency(records: list[PredictionRecord]) -> dict:
    latencies = [r.latency_s for r in records if r.latency_s is not None]
    costs = [r.cost_usd for r in records if r.cost_usd is not None]
    out = {}
    if latencies:
        out["latency_p50_s"] = float(np.percentile(latencies, 50))
        out["latency_p95_s"] = float(np.percentile(latencies, 95))
    if costs:
        out["usd_per_1k_reviews"] = 1000 * float(np.mean(costs))
    return out


def evaluate(
    predictions: dict[str, PredictionRecord],
    gold: dict[str, Pairs],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    ids = sorted(gold)
    pred = {rid: predictions[rid].pairs() for rid in ids if rid in predictions}
    present = [predictions[rid] for rid in ids if rid in predictions]
    n_evidence = sum(r.n_evidence for r in present)

    metrics = compute_all(gold, pred, ids)
    cis = {
        name: bootstrap_ci(ids, lambda s, n=name: compute_all(gold, pred, s)[n], n_boot, seed)
        for name in HEADLINE
    }
    return {
        "n_gold": len(ids),
        "n_predicted": len(present),
        "valid_json_rate": sum(r.valid_json for r in present) / len(ids),
        "valid_schema_rate": sum(r.valid_schema for r in present) / len(ids),
        "repaired_rate": sum(r.repaired for r in present) / len(ids),
        "usable_rate": sum(r.analysis is not None or r.mentions is not None for r in present)
        / len(ids),
        "grounding_rate": (sum(r.n_grounded for r in present) / n_evidence) if n_evidence else None,
        **{k: v for k, v in metrics.items() if k != "detail"},
        "ci95": cis,
        "efficiency": efficiency(present),
        "detail": metrics["detail"],
    }


def run_evaluation(
    predictions_file: Path,
    annotations_dir: Path,
    results_dir: Path,
    split: str,
    system: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    gold = load_gold(annotations_dir, split)
    result = {
        "system": system,
        "split": split,
        **evaluate(read_predictions(predictions_file), gold, n_boot=n_boot, seed=seed),
    }
    out = results_dir / "metrics" / f"{system}__{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if split == "gold_test":
        with (results_dir / "test_access.log").open("a", encoding="utf-8") as log:
            log.write(f"{datetime.now(UTC).isoformat(timespec='seconds')}\t{system}\n")
    return result


def compare(
    file_a: Path, file_b: Path, annotations_dir: Path, split: str, metric: str, n_boot: int = 1000
) -> dict:
    gold = load_gold(annotations_dir, split)
    ids = sorted(gold)
    a, b = read_predictions(file_a), read_predictions(file_b)
    pa = {rid: a[rid].pairs() for rid in ids if rid in a}
    pb = {rid: b[rid].pairs() for rid in ids if rid in b}
    return paired_bootstrap(
        ids,
        lambda s: compute_all(gold, pa, s)[metric],
        lambda s: compute_all(gold, pb, s)[metric],
        n_boot=n_boot,
    )


def _fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}" if pct else f"{x:.2f}"


def report(results_dir: Path, split: str) -> str:
    """Markdown comparison table from every `results/metrics/*__{split}.json`."""
    rows = []
    for path in sorted((results_dir / "metrics").glob(f"*__{split}.json")):
        r = json.loads(path.read_text())
        ci = r["ci95"]["aspect_macro_f1"]
        eff = r.get("efficiency", {})
        rows.append(
            f"| {r['system']} | {_fmt(r['valid_schema_rate'])} "
            f"| {_fmt(r['aspect_macro_f1'])} [{_fmt(ci['ci_low'])}–{_fmt(ci['ci_high'])}] "
            f"| {_fmt(r['joint_macro_f1'])} | {_fmt(r['sentiment_macro_f1_matched'])} "
            f"| {_fmt(r['grounding_rate'])} | {_fmt(eff.get('latency_p50_s'), pct=False)} "
            f"| {_fmt(eff.get('usd_per_1k_reviews'), pct=False)} |"
        )
    header = (
        f"Results on {split} (n = gold reviews; F1 in %, 95% bootstrap CI)\n\n"
        "| System | Valid JSON | Aspect macro-F1 | Joint macro-F1 | Sentiment F1 (matched) "
        "| Evidence grounded | Latency p50 (s) | US$/1k reviews |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    table = "\n".join([header, *rows]) + "\n"
    (results_dir / f"comparison_{split}.md").write_text(table)
    return table
