"""One prediction format for every system (teacher, Claude, BERTimbau, student), so a single
evaluation script can score them all on the same gold set."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from pitaco.parsing import parse_output
from pitaco.schema import GoldMention, ReviewAnalysis


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    system: str
    raw: str | None  # exact model output; None for classifiers that emit labels directly
    analysis: ReviewAnalysis | None
    # Label-only systems (BERTimbau) emit no JSON, evidence or summary — only aspect + sentiment.
    mentions: list[GoldMention] | None = None
    valid_json: bool
    valid_schema: bool
    n_evidence: int = 0
    n_grounded: int = 0
    latency_s: float | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None

    def pairs(self) -> set[tuple[str, str]]:
        if self.analysis is not None:
            return self.analysis.pairs()
        return {(m.aspecto, m.sentimento) for m in self.mentions or []}


def record_from_raw(
    review_id: str, system: str, raw: str, review_text: str, fuzzy_threshold: int = 90, **meta
) -> PredictionRecord:
    parsed = parse_output(raw, review_text, fuzzy_threshold)
    return PredictionRecord(
        review_id=review_id,
        system=system,
        raw=raw,
        analysis=parsed.analysis,
        valid_json=parsed.valid_json,
        valid_schema=parsed.valid_schema,
        n_evidence=parsed.n_evidence,
        n_grounded=parsed.n_grounded,
        error=parsed.error,
        **meta,
    )


def write_predictions(path: Path, records: list[PredictionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in sorted(records, key=lambda r: r.review_id):
            f.write(record.model_dump_json() + "\n")


def read_predictions(path: Path) -> dict[str, PredictionRecord]:
    records = (
        PredictionRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {r.review_id: r for r in records}
