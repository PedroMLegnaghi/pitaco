import json

from pitaco.eval.evaluate import report, run_evaluation
from pitaco.labeling.annotations import append_annotation, build_annotation
from pitaco.predictions import record_from_raw, write_predictions

TEXT = {"r1": "Chegou antes do prazo, adorei", "r2": "Veio quebrado"}


def _setup(tmp_path):
    ann_dir = tmp_path / "annotations"
    for rid, choices in [("r1", {"entrega": "positivo"}), ("r2", {"produto": "negativo"})]:
        b = build_annotation(rid, choices, difficult=False, note="", annotator="t")
        append_annotation(ann_dir / "gold_test.jsonl", b)
    out = {
        "aspectos": [
            {"aspecto": "entrega", "sentimento": "positivo", "evidencia": "Chegou antes do prazo"}
        ],
        "resumo": "Entrega antecipada.",
    }
    preds = [
        record_from_raw("r1", "sys", json.dumps(out), TEXT["r1"], latency_s=1.0, cost_usd=0.002),
        record_from_raw("r2", "sys", "not json", TEXT["r2"], latency_s=3.0, cost_usd=0.002),
    ]
    pred_file = tmp_path / "preds.jsonl"
    write_predictions(pred_file, preds)
    return ann_dir, pred_file


def test_end_to_end_evaluation_and_report(tmp_path):
    ann_dir, pred_file = _setup(tmp_path)
    results = tmp_path / "results"
    r = run_evaluation(pred_file, ann_dir, results, "gold_test", "sys", n_boot=50)
    assert r["n_gold"] == 2 and r["valid_schema_rate"] == 0.5
    assert r["grounding_rate"] == 1.0
    assert r["efficiency"]["usd_per_1k_reviews"] == 2.0
    assert r["efficiency"]["latency_p50_s"] == 2.0
    assert (results / "metrics" / "sys__gold_test.json").exists()
    assert "sys" in (results / "test_access.log").read_text()  # test use is logged
    table = report(results, "gold_test")
    assert "| sys |" in table and (results / "comparison_gold_test.md").exists()


def test_dev_evaluation_is_not_logged_as_test_access(tmp_path):
    ann_dir, pred_file = _setup(tmp_path)
    (ann_dir / "gold_test.jsonl").rename(ann_dir / "gold_dev.jsonl")
    results = tmp_path / "results"
    run_evaluation(pred_file, ann_dir, results, "gold_dev", "sys", n_boot=20)
    assert not (results / "test_access.log").exists()
