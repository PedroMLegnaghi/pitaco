import json

import pytest

from pitaco.parsing import extract_json_object, locate_evidence, parse_output
from pitaco.schema import strict_json_schema

REVIEW = "Recebi no prazo, porém a Cor veio diferente do anúncio. Péssimo!"
GOOD = {
    "aspectos": [
        {"aspecto": "entrega", "sentimento": "positivo", "evidencia": "Recebi no prazo"},
        {"aspecto": "produto", "sentimento": "negativo", "evidencia": "a Cor veio diferente"},
    ],
    "resumo": "Entrega no prazo, mas cor diferente.",
}


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps(GOOD),
        "```json\n" + json.dumps(GOOD) + "\n```",
        "Claro! Aqui está:\n" + json.dumps(GOOD) + "\nEspero ter ajudado.",
    ],
)
def test_extract_tolerates_fences_and_chatter(raw):
    assert extract_json_object(raw) == GOOD


@pytest.mark.parametrize("raw", ["", "não sei", "{aspectos: []}", "[1, 2]"])
def test_extract_returns_none_on_garbage(raw):
    assert extract_json_object(raw) is None


def test_valid_and_fully_grounded():
    out = parse_output(json.dumps(GOOD), REVIEW)
    assert out.valid_json and out.valid_schema and out.fully_grounded
    assert out.n_exact == 2
    assert out.analysis.pairs() == {("entrega", "positivo"), ("produto", "negativo")}


def test_invalid_schema_is_reported_not_raised():
    bad = {**GOOD, "aspectos": [{**GOOD["aspectos"][0], "sentimento": "misto"}]}
    out = parse_output(json.dumps(bad), REVIEW)
    assert out.valid_json and not out.valid_schema and out.analysis is None
    assert out.error


def test_no_json():
    out = parse_output("desculpe", REVIEW)
    assert not out.valid_json and not out.valid_schema


@pytest.mark.parametrize(
    ("evidence", "expected_span", "how"),
    [
        ("Recebi no prazo", "Recebi no prazo", "exact"),
        ("recebi  no PRAZO", "Recebi no prazo", "normalized"),
        ("a cor veio diferente do anuncio", "a Cor veio diferente do anúncio", "fuzzy"),
        ("produto excelente e barato", None, "none"),
    ],
)
def test_locate_evidence(evidence, expected_span, how):
    assert locate_evidence(evidence, REVIEW) == (expected_span, how)


def test_snapped_evidence_is_copied_from_the_review():
    paraphrased = {**GOOD}
    paraphrased["aspectos"] = [{**GOOD["aspectos"][0], "evidencia": "recebi no PRAZO"}]
    out = parse_output(json.dumps(paraphrased), REVIEW)
    assert out.analysis.aspectos[0].evidencia == "Recebi no prazo"
    assert out.n_snapped == 1 and out.fully_grounded


def test_hallucinated_evidence_counts_as_ungrounded():
    invented = {**GOOD}
    invented["aspectos"] = [{**GOOD["aspectos"][0], "evidencia": "entregador muito educado"}]
    out = parse_output(json.dumps(invented), REVIEW)
    assert out.valid_schema and not out.fully_grounded and out.n_ungrounded == 1


def test_strict_schema_is_self_contained_and_closed():
    schema = strict_json_schema()
    text = json.dumps(schema)
    assert "$ref" not in text and "$defs" not in text and "minLength" not in text
    assert schema["additionalProperties"] is False
    item = schema["properties"]["aspectos"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"aspecto", "sentimento", "evidencia"}
    assert "entrega" in item["properties"]["aspecto"]["enum"]
