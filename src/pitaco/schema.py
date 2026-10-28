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


class AspectMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aspecto: Aspect
    sentimento: Sentiment
    evidencia: str = Field(min_length=1, description="Verbatim excerpt from the review")


class ReviewAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aspectos: list[AspectMention]
    resumo: str = Field(min_length=1, description="One-sentence summary in Portuguese")

    @field_validator("aspectos")
    @classmethod
    def one_entry_per_aspect(cls, mentions: list[AspectMention]) -> list[AspectMention]:
        # Annotation rule: an aspect gets a single, dominant polarity per review.
        # This keeps every metric a clean per-aspect comparison.
        seen = [m.aspecto for m in mentions]
        duplicates = sorted({a for a in seen if seen.count(a) > 1})
        if duplicates:
            raise ValueError(f"aspect listed more than once: {', '.join(duplicates)}")
        return mentions

    def pairs(self) -> set[tuple[str, str]]:
        """(aspect, sentiment) pairs — the unit the metrics compare."""
        return {(m.aspecto, m.sentimento) for m in self.aspectos}
