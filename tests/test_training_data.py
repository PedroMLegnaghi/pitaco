import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitaco.models.bert_labels import LABELS, decode, encode, tune_threshold
from pitaco.models.training_data import build, compact_json, keep, nested_subsets
from pitaco.predictions import record_from_raw, write_predictions
from pitaco.schema import ReviewAnalysis

PROMPTS = Path(__file__).parents[1] / "prompts"


def test_label_space():
    assert len(LABELS) == 18 and LABELS[0] == "entrega:positivo"


def test_encode_decode_round_trip():
    pairs = {("entrega", "negativo"), ("preco", "positivo")}
    assert decode(encode(pairs), threshold=0.5) == pairs


def test_decode_keeps_one_sentiment_per_aspect_and_applies_threshold():
    probs = np.zeros(18)
    probs[0:3] = [0.6, 0.1, 0.7]  # entrega: positivo 0.6, negativo 0.7 → negativo wins
    probs[3:6] = [0.3, 0.0, 0.0]  # produto: below threshold → absent
    assert decode(probs, threshold=0.5) == {("entrega", "negativo")}


def test_threshold_tuning_finds_the_separating_value():
    gold = [{("entrega", "positivo")}, set()]
    probs = np.zeros((2, 18))
    probs[0, 0] = 0.8  # true label
    probs[1, 0] = 0.4  # false alarm — a threshold above 0.4 removes it
    threshold, score = tune_threshold(probs, gold)
    assert 0.4 < threshold <= 0.8 and score == 1.0


def test_compact_json_is_parseable_and_unescaped():
    analysis = ReviewAnalysis.model_validate(
        {
            "aspectos": [
                {"aspecto": "preco", "sentimento": "positivo", "evidencia": "preço ótimo"}
            ],
            "resumo": "Preço ótimo.",
        }
    )
    text = compact_json(analysis)
    assert "preço ótimo" in text and ", " not in text
    assert ReviewAnalysis.model_validate_json(text) == analysis


def test_nested_subsets_are_nested_and_stratified():
    df = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(500)],
            "review_score": [1 + i % 5 for i in range(500)],
        }
    )
    subsets = nested_subsets(df, [50, 150], seed=0)
    assert subsets[50].review_score.value_counts().to_dict() == {s: 10 for s in range(1, 6)}
    assert set(subsets[50].review_id) <= set(subsets[150].review_id)


def _write_split(tmp_path, split, rows):
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_json(
        processed / f"{split}.jsonl", orient="records", lines=True, force_ascii=False
    )


def test_build_filters_ungrounded_and_writes_both_formats(tmp_path):
    good = {
        "aspectos": [
            {"aspecto": "entrega", "sentimento": "positivo", "evidencia": "chegou rápido"}
        ],
        "resumo": "Rápido.",
    }
    invented = {
        "aspectos": [
            {"aspecto": "produto", "sentimento": "positivo", "evidencia": "qualidade incrível"}
        ],
        "resumo": "x",
    }
    rows, preds = [], []
    for i in range(10):
        rid, text = f"r{i}", "chegou rápido demais"
        rows.append(
            {
                "review_id": rid,
                "review_score": 1 + i % 5,
                "text": text,
                "review_creation_date": "2018-01-01",
            }
        )
        out = good if i < 8 else invented
        preds.append(record_from_raw(rid, "t", json.dumps(out, ensure_ascii=False), text))
    for split in ("silver_train", "silver_val"):
        _write_split(tmp_path, split, rows)
        write_predictions(tmp_path / "preds" / f"{split}.jsonl", preds)

    report = build(
        tmp_path / "preds", tmp_path / "processed", tmp_path / "out", PROMPTS, "v1", [5], seed=0
    )
    assert report["silver_train"]["kept"] == 8  # the 2 ungrounded outputs are dropped
    line = json.loads((tmp_path / "out" / "train_all" / "sft.jsonl").read_text().splitlines()[0])
    assert [m["role"] for m in line["messages"]] == ["system", "user", "assistant"]
    assert ReviewAnalysis.model_validate_json(line["messages"][2]["content"])
    bert = json.loads((tmp_path / "out" / "val" / "bert.jsonl").read_text().splitlines()[0])
    assert len(bert["labels"]) == 18 and sum(bert["labels"]) == 1


@pytest.mark.parametrize(
    ("valid", "grounded", "expected"), [(True, 1, True), (True, 0, False), (False, 0, False)]
)
def test_keep_rule(valid, grounded, expected):
    out = {
        "aspectos": [{"aspecto": "entrega", "sentimento": "positivo", "evidencia": "chegou"}],
        "resumo": "ok",
    }
    text = "chegou" if grounded else "nada a ver"
    raw = json.dumps(out) if valid else "x"
    assert keep(record_from_raw("r", "s", raw, text)) is expected
