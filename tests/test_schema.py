import pytest
from pydantic import ValidationError

from pitaco.schema import ASPECTS, SENTIMENTS, ReviewAnalysis

VALID = {
    "aspectos": [
        {"aspecto": "entrega", "sentimento": "positivo", "evidencia": "chegou antes do prazo"},
        {"aspecto": "produto", "sentimento": "negativo", "evidencia": "veio com defeito"},
    ],
    "resumo": "Entrega rápida, mas o produto veio com defeito.",
}


def test_taxonomy():
    assert ASPECTS == ("entrega", "produto", "embalagem", "atendimento", "preco", "outro")
    assert SENTIMENTS == ("positivo", "neutro", "negativo")


def test_valid_output_parses_and_exposes_pairs():
    analysis = ReviewAnalysis.model_validate(VALID)
    assert analysis.pairs() == {("entrega", "positivo"), ("produto", "negativo")}


def test_empty_aspect_list_is_valid():
    ReviewAnalysis.model_validate({"aspectos": [], "resumo": "Sem aspectos claros."})


@pytest.mark.parametrize(
    ("field", "value"),
    [("aspecto", "frete"), ("sentimento", "misto"), ("evidencia", "")],
)
def test_invalid_mention_is_rejected(field, value):
    bad = {"aspectos": [{**VALID["aspectos"][0], field: value}], "resumo": "x"}
    with pytest.raises(ValidationError):
        ReviewAnalysis.model_validate(bad)


def test_extra_keys_are_rejected():
    with pytest.raises(ValidationError):
        ReviewAnalysis.model_validate({**VALID, "nota": 5})


def test_duplicate_aspect_is_rejected():
    dup = {"aspectos": [VALID["aspectos"][0], VALID["aspectos"][0]], "resumo": "x"}
    with pytest.raises(ValidationError, match="more than once"):
        ReviewAnalysis.model_validate(dup)
