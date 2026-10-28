import pytest
from pydantic import ValidationError

from pitaco.labeling.annotations import (
    ABSENT,
    annotation_order,
    append_annotation,
    build_annotation,
    choices_from_annotation,
    load_annotations,
)
from pitaco.schema import ASPECTS


def _ann(rid="r1", **choices):
    return build_annotation(rid, choices, difficult=False, note=" ", annotator="pedro")


def test_order_is_deterministic_and_not_file_order():
    ids = [f"r{i:02d}" for i in range(50)]
    assert annotation_order(ids, 42) == annotation_order(list(reversed(ids)), 42)
    assert annotation_order(ids, 42) != ids
    assert annotation_order(ids, 42) != annotation_order(ids, 7)


def test_build_drops_absent_aspects_and_strips_note():
    ann = _ann(entrega="negativo", produto="positivo", preco=ABSENT)
    assert ann.pairs() == {("entrega", "negativo"), ("produto", "positivo")}
    assert ann.note == ""


def test_choices_round_trip():
    ann = _ann(embalagem="neutro")
    choices = choices_from_annotation(ann)
    assert set(choices) == set(ASPECTS)
    assert choices["embalagem"] == "neutro"
    assert choices["entrega"] == ABSENT
    assert choices_from_annotation(None) == {a: ABSENT for a in ASPECTS}


def test_invalid_sentiment_or_aspect_rejected():
    with pytest.raises(ValidationError):
        _ann(entrega="misto")
    with pytest.raises(ValueError, match="unknown"):
        _ann(frete="negativo")


def test_relabel_appends_and_latest_wins(tmp_path):
    path = tmp_path / "gold.jsonl"
    append_annotation(path, _ann(entrega="positivo"))
    append_annotation(path, _ann("r2", produto="negativo"))
    append_annotation(path, _ann(entrega="negativo"))
    loaded = load_annotations(path)
    assert len(path.read_text().splitlines()) == 3  # audit trail kept
    assert loaded["r1"].pairs() == {("entrega", "negativo")}
    assert set(loaded) == {"r1", "r2"}


def test_missing_file_means_no_annotations(tmp_path):
    assert load_annotations(tmp_path / "none.jsonl") == {}
