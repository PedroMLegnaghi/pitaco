"""The few-shot examples teach the teacher by example — a broken example is a systematic error."""

from pathlib import Path

import pandas as pd
import pytest

from pitaco.labeling.prompts import load_fewshot, student_instruction, teacher_system_prompt
from pitaco.schema import ReviewAnalysis

ROOT = Path(__file__).parents[1]
PROMPTS = ROOT / "prompts"
EXAMPLES = load_fewshot(PROMPTS, "v1")


@pytest.mark.parametrize("ex", EXAMPLES, ids=lambda ex: ex["review_id"][:8])
def test_fewshot_output_is_valid_and_evidence_is_verbatim(ex):
    analysis = ReviewAnalysis.model_validate(ex["output"])
    for mention in analysis.aspectos:
        assert mention.evidencia in ex["text"], mention.evidencia


def test_fewshot_covers_every_aspect_and_sentiment():
    pairs = set().union(*(ReviewAnalysis.model_validate(ex["output"]).pairs() for ex in EXAMPLES))
    assert {a for a, _ in pairs} == {
        "entrega",
        "produto",
        "embalagem",
        "atendimento",
        "preco",
        "outro",
    }
    assert {s for _, s in pairs} == {"positivo", "neutro", "negativo"}


@pytest.mark.skipif(
    not (ROOT / "data/processed/gold_test.jsonl").exists(), reason="splits not built"
)
def test_fewshot_examples_come_from_silver_and_match_its_text():
    processed = ROOT / "data/processed"
    gold = pd.concat(
        pd.read_json(processed / f, lines=True) for f in ("gold_dev.jsonl", "gold_test.jsonl")
    )
    silver = pd.read_json(processed / "silver_train.jsonl", lines=True).set_index("review_id")
    for ex in EXAMPLES:
        assert ex["review_id"] not in set(gold.review_id)
        assert silver.loc[ex["review_id"], "text"] == ex["text"]


def test_zero_and_few_shot_share_the_same_instructions():
    zero = teacher_system_prompt(PROMPTS, "v1", "zeroshot")
    few = teacher_system_prompt(PROMPTS, "v1", "fewshot")
    assert few.startswith(zero) and "Exemplos:" in few and "Exemplos:" not in zero


def test_student_instruction_is_short():
    assert len(student_instruction(PROMPTS, "v1")) < 500  # it is paid on every inference
