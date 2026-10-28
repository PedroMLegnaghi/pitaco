"""Turn raw model text into a validated `ReviewAnalysis`, and check the evidence is really quoted.

Two separate questions, reported separately:
- validity: is the output parseable JSON that matches the schema?
- grounding: is every `evidencia` an actual span of the review? A paraphrased or invented span is
  a hallucination even when the aspect and sentiment are right.
"""

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError
from rapidfuzz import fuzz

from pitaco.schema import AspectMention, ReviewAnalysis

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json_object(text: str) -> dict | None:
    """Parse a JSON object from model output, tolerating ```json fences and text around it."""
    cleaned = _FENCE.sub("", text.strip())
    for candidate in (cleaned, cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def locate_evidence(evidence: str, text: str, fuzzy_threshold: int = 90) -> tuple[str | None, str]:
    """Find `evidence` in `text`. Returns (span copied from the text, how it was found).

    exact      → verbatim substring
    normalized → same words, different case/whitespace; snapped to the original span
    fuzzy      → near match (e.g. a fixed typo) with partial ratio >= threshold; snapped
    none       → not grounded
    """
    if evidence and evidence in text:
        return evidence, "exact"
    tokens = evidence.split()
    if tokens:
        pattern = r"\s+".join(map(re.escape, tokens))
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0), "normalized"
    if len(evidence) >= 4:
        aligned = fuzz.partial_ratio_alignment(evidence.lower(), text.lower())
        if aligned is not None and aligned.score >= fuzzy_threshold:
            start, end = aligned.dest_start, aligned.dest_end
            while start > 0 and text[start - 1].isalnum():
                start -= 1
            while end < len(text) and text[end].isalnum():
                end += 1
            span = text[start:end].strip()
            if span:
                return span, "fuzzy"
    return None, "none"


@dataclass(frozen=True)
class ParsedOutput:
    valid_json: bool
    valid_schema: bool
    analysis: ReviewAnalysis | None  # evidence snapped to the review text where possible
    error: str | None = None
    n_evidence: int = 0
    n_exact: int = 0
    n_snapped: int = 0
    n_ungrounded: int = 0

    @property
    def n_grounded(self) -> int:
        return self.n_exact + self.n_snapped

    @property
    def fully_grounded(self) -> bool:
        return self.valid_schema and self.n_ungrounded == 0


def parse_output(raw: str, review_text: str, fuzzy_threshold: int = 90) -> ParsedOutput:
    obj = extract_json_object(raw)
    if obj is None:
        return ParsedOutput(valid_json=False, valid_schema=False, analysis=None, error="no JSON")
    try:
        analysis = ReviewAnalysis.model_validate(obj)
    except ValidationError as exc:
        error = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        return ParsedOutput(valid_json=True, valid_schema=False, analysis=None, error=error)

    mentions, exact, snapped, ungrounded = [], 0, 0, 0
    for mention in analysis.aspectos:
        span, how = locate_evidence(mention.evidencia, review_text, fuzzy_threshold)
        if how == "exact":
            exact += 1
        elif span is not None:
            snapped += 1
        else:
            ungrounded += 1
        mentions.append(
            AspectMention(
                aspecto=mention.aspecto,
                sentimento=mention.sentimento,
                evidencia=span or mention.evidencia,
            )
        )
    return ParsedOutput(
        valid_json=True,
        valid_schema=True,
        analysis=ReviewAnalysis(aspectos=mentions, resumo=analysis.resumo),
        n_evidence=len(mentions),
        n_exact=exact,
        n_snapped=snapped,
        n_ungrounded=ungrounded,
    )
