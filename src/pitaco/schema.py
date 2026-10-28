"""Output schema: the single source of truth for the taxonomy and the JSON every system must emit.

The teacher's labels, the student's outputs, the gold annotations and the metrics all validate
against these models, so a change here propagates everywhere.
"""

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

Aspect = Literal["entrega", "produto", "embalagem", "atendimento", "preco", "outro"]
Sentiment = Literal["positivo", "neutro", "negativo"]

ASPECTS: tuple[str, ...] = get_args(Aspect)
SENTIMENTS: tuple[str, ...] = get_args(Sentiment)


def _check_unique_aspects(mentions: list) -> list:
    # Annotation rule: an aspect gets a single, dominant polarity per review.
    # This keeps every metric a clean per-aspect comparison.
    seen = [m.aspecto for m in mentions]
    duplicates = sorted({a for a in seen if seen.count(a) > 1})
    if duplicates:
        raise ValueError(f"aspect listed more than once: {', '.join(duplicates)}")
    return mentions


class GoldMention(BaseModel):
    """A hand label: aspect + sentiment only (evidence is scored automatically, not annotated)."""

    model_config = ConfigDict(extra="forbid")

    aspecto: Aspect
    sentimento: Sentiment


class AspectMention(GoldMention):
    model_config = ConfigDict(extra="forbid")

    evidencia: str = Field(min_length=1, description="Verbatim excerpt from the review")


class ReviewAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aspectos: list[AspectMention]
    resumo: str = Field(min_length=1, description="One-sentence summary in Portuguese")

    @field_validator("aspectos")
    @classmethod
    def one_entry_per_aspect(cls, mentions: list[AspectMention]) -> list[AspectMention]:
        return _check_unique_aspects(mentions)

    def pairs(self) -> set[tuple[str, str]]:
        """(aspect, sentiment) pairs — the unit the metrics compare."""
        return {(m.aspecto, m.sentimento) for m in self.aspectos}


class GoldAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    aspectos: list[GoldMention]
    difficult: bool = False
    note: str = ""
    annotator: str
    labeled_at: str

    @field_validator("aspectos")
    @classmethod
    def one_entry_per_aspect(cls, mentions: list[GoldMention]) -> list[GoldMention]:
        return _check_unique_aspects(mentions)

    def pairs(self) -> set[tuple[str, str]]:
        return {(m.aspecto, m.sentimento) for m in self.aspectos}


def strict_json_schema() -> dict:
    """JSON schema of `ReviewAnalysis` for constrained decoding (`response_format: json_schema`).

    Inlines `$defs` (several providers reject `$ref`) and drops keywords that strict modes commonly
    refuse (`title`, `minLength`). Enums and `additionalProperties: false` are kept — they are what
    makes constrained decoding useful here.
    """
    schema = ReviewAnalysis.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: resolve(v) for k, v in node.items() if k not in ("title", "minLength")}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)
