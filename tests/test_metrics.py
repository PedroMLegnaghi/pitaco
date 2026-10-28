"""Metrics checked against values computed by hand. If these are wrong, every number is wrong."""

import pytest

from pitaco.eval.bootstrap import bootstrap_ci, paired_bootstrap
from pitaco.eval.metrics import (
    aspect_detection,
    cohen_kappa,
    compute_all,
    joint_pairs,
    prf,
    sentiment_on_matched,
)

# Three reviews:
#   r1 gold {entrega+, produto-}   pred {entrega+, produto+}   (produto: wrong polarity)
#   r2 gold {embalagem-}           pred {}                     (missed)
#   r3 gold {preco+}               pred {preco+, outro+}       (one extra aspect)
GOLD = {
    "r1": {("entrega", "positivo"), ("produto", "negativo")},
    "r2": {("embalagem", "negativo")},
    "r3": {("preco", "positivo")},
}
PRED = {
    "r1": {("entrega", "positivo"), ("produto", "positivo")},
    "r3": {("preco", "positivo"), ("outro", "positivo")},
}
IDS = ["r1", "r2", "r3"]


def test_prf_edge_cases():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert prf(1, 1, 0) == pytest.approx((0.5, 1.0, 2 / 3))


def test_aspect_detection_by_hand():
    d = aspect_detection(GOLD, PRED, IDS)
    pa = d["per_aspect"]
    assert pa["entrega"]["f1"] == 1.0
    assert pa["produto"]["f1"] == 1.0  # detection ignores polarity
    assert pa["embalagem"]["f1"] == 0.0 and pa["embalagem"]["support"] == 1
    assert pa["outro"]["f1"] == 0.0 and pa["outro"]["predicted"] == 1
    assert pa["atendimento"]["support"] == 0 and pa["atendimento"]["predicted"] == 0
    # macro over the 5 defined aspects (atendimento undefined): (1 + 1 + 0 + 1 + 0) / 5
    assert d["macro_f1"] == pytest.approx(3 / 5)
    # micro: tp=3 (entrega, produto, preco), fp=1 (outro), fn=1 (embalagem) → P=R=0.75
    assert d["micro_f1"] == pytest.approx(0.75)


def test_joint_pairs_by_hand():
    j = joint_pairs(GOLD, PRED, IDS)
    assert j["per_aspect"]["produto"]["f1"] == 0.0  # wrong polarity = 1 FP + 1 FN
    assert j["per_aspect"]["entrega"]["f1"] == 1.0
    # tp=2 (entrega, preco), fp=2 (produto+, outro), fn=2 (produto-, embalagem) → 0.5
    assert j["micro_f1"] == pytest.approx(0.5)
    assert j["macro_f1"] == pytest.approx(2 / 5)


def test_sentiment_on_matched_by_hand():
    s = sentiment_on_matched(GOLD, PRED, IDS)
    assert s["n"] == 3  # entrega, produto, preco found by both
    assert s["accuracy"] == pytest.approx(2 / 3)
    # positivo: tp=2 fp=1 fn=0 → F1 0.8; negativo: tp=0 fp=0 fn=1 → 0; neutro absent → skipped
    assert s["macro_f1"] == pytest.approx(0.4)


def test_cohen_kappa_known_values():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    # observed 0.5, expected 0.5 → 0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0


def test_perfect_prediction_scores_one():
    m = compute_all(GOLD, GOLD, IDS)
    assert m["aspect_macro_f1"] == m["joint_macro_f1"] == m["joint_micro_f1"] == 1.0
    assert m["kappa_mean"] == 1.0


def test_duplicated_ids_count_twice():
    once = aspect_detection(GOLD, PRED, ["r2"])
    twice = aspect_detection(GOLD, PRED, ["r2", "r2"])
    assert (
        twice["per_aspect"]["embalagem"]["support"]
        == 2 * once["per_aspect"]["embalagem"]["support"]
    )


def test_bootstrap_ci_contains_point_estimate():
    ids = [f"r{i}" for i in range(40)]
    gold = {rid: {("entrega", "positivo")} for rid in ids}
    pred = {rid: ({("entrega", "positivo")} if i % 4 else set()) for i, rid in enumerate(ids)}
    ci = bootstrap_ci(ids, lambda s: compute_all(gold, pred, s)["joint_micro_f1"], n_boot=200)
    assert ci["ci_low"] <= ci["value"] <= ci["ci_high"]
    assert ci["ci_high"] - ci["ci_low"] > 0


def test_paired_bootstrap_detects_a_clearly_better_system():
    ids = [f"r{i}" for i in range(60)]
    gold = {rid: {("produto", "negativo")} for rid in ids}
    good = {rid: {("produto", "negativo")} for rid in ids}
    bad = {rid: set() for rid in ids}
    res = paired_bootstrap(
        ids,
        lambda s: compute_all(gold, good, s)["joint_micro_f1"],
        lambda s: compute_all(gold, bad, s)["joint_micro_f1"],
        n_boot=200,
    )
    assert res["diff"] == 1.0 and res["ci_low"] > 0 and res["p_b_not_worse"] == 0.0
