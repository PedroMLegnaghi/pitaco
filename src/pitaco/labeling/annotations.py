"""Storage and ordering for hand labels (no UI code, so it is testable without Gradio).

Labels are appended to a JSONL file, one record per save. Re-labeling a review appends a new
record and the latest one wins on load: a crash never loses earlier work, and the file keeps an
audit trail.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from pitaco.schema import ASPECTS, GoldAnnotation, GoldMention

ABSENT = "—"
CHOICES = (ABSENT, "positivo", "neutro", "negativo")


def annotation_order(review_ids: list[str], seed: int) -> list[str]:
    """Deterministic pseudo-random order.

    The split files are grouped by star rating; showing them in file order would leak the rating
    (ten 1-star reviews in a row), which is exactly the signal we hide from the annotator.
    """
    return sorted(review_ids, key=lambda rid: hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest())


def load_annotations(path: Path) -> dict[str, GoldAnnotation]:
    if not path.exists():
        return {}
    latest: dict[str, GoldAnnotation] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ann = GoldAnnotation.model_validate_json(line)
            latest[ann.review_id] = ann
    return latest


def append_annotation(path: Path, ann: GoldAnnotation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(ann.model_dump_json() + "\n")


def build_annotation(
    review_id: str, choices: dict[str, str], *, difficult: bool, note: str, annotator: str
) -> GoldAnnotation:
    """Turn the UI state (one radio per aspect) into a validated annotation."""
    unknown = set(choices) - set(ASPECTS)
    if unknown:
        raise ValueError(f"unknown aspects: {sorted(unknown)}")
    mentions = [
        GoldMention(aspecto=aspect, sentimento=choices[aspect])
        for aspect in ASPECTS
        if choices.get(aspect, ABSENT) != ABSENT
    ]
    return GoldAnnotation(
        review_id=review_id,
        aspectos=mentions,
        difficult=difficult,
        note=note.strip(),
        annotator=annotator,
        labeled_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def choices_from_annotation(ann: GoldAnnotation | None) -> dict[str, str]:
    """Inverse of `build_annotation`, used to pre-fill the radios when revisiting a review."""
    chosen = {m.aspecto: m.sentimento for m in ann.aspectos} if ann else {}
    return {aspect: chosen.get(aspect, ABSENT) for aspect in ASPECTS}


def export_rows(annotations: dict[str, GoldAnnotation]) -> list[dict]:
    """Flat records for downstream stages, sorted by review_id."""
    return [json.loads(annotations[rid].model_dump_json()) for rid in sorted(annotations)]
