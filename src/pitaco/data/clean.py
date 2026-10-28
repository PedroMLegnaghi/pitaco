"""Text normalization and PII scrubbing (LGPD data minimization).

Olist already pseudonymized store names (they appear as Game of Thrones houses, e.g. "lannister"),
but customers sometimes write phone numbers or sign their names. We replace those spans with
placeholders before any text leaves the laptop.

Names are found with targeted rules, not NER: spaCy's Portuguese model flagged ~11% of reviews as
containing a person, almost all false positives on capitalized verbs ("Comprei", "Amei"), while
missing lowercase signatures. Rules are narrower but precise; residual risk is documented in the
dataset card.
"""

import re
from collections import Counter

_UPPER_WORD = r"[A-ZÀ-Ý][a-zà-ÿ]+"
_ANY_WORD = r"[A-Za-zÀ-ÿ]+"

# Order matters: longer numeric patterns (CNPJ, CPF) run before the phone pattern.
PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("url", re.compile(r"(?i)\b(?:https?://|www\.)\S+"), "[URL]"),
    ("cnpj", re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"), "[CNPJ]"),
    ("cpf", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b"), "[CPF]"),
    ("cep", re.compile(r"\b\d{5}-\d{3}\b"), "[CEP]"),
    (
        "phone",
        re.compile(r"(?:\+?55\s?)?\(?\b\d{2}\)?[\s.-]?9?\s?\d{4}[\s.-]?\d{4}\b"),
        "[TELEFONE]",
    ),
    # "Sr. Bruno", "Sra Maria Souza" — honorific kept, name replaced. Requires a capital letter,
    # so the common typo "parece Sr muito boa" (for "ser") is left alone.
    (
        "name_honorific",
        re.compile(
            rf"\b((?:[Ss]r|[Ss]ra|[Ss]rta|[Dd]r|[Dd]ra)\.?\s+){_UPPER_WORD}(?:\s+{_UPPER_WORD})*"
        ),
        r"\1[NOME]",
    ),
    # "meu nome é Ana", "me chamo João Silva" — extra words count only if capitalized (surnames)
    (
        "name_intro",
        re.compile(rf"\b([Mm]eu nome [ée]|[Mm]e chamo)\s+{_ANY_WORD}(?:\s+{_UPPER_WORD})*"),
        r"\1 [NOME]",
    ),
    # Sign-off at the very end: "Att Elenice.", "atenciosamente, alice bretanha"
    (
        "name_signoff",
        re.compile(
            rf"(?i)\b(att\.?|atenciosamente|ass\.?)[\s,:-]+{_ANY_WORD}(?:\s+{_ANY_WORD}){{0,2}}[\s.!]*$"
        ),
        r"\1 [NOME]",
    ),
]

_WHITESPACE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse \\r\\n, tabs and repeated spaces into single spaces."""
    return _WHITESPACE.sub(" ", text).strip()


def scrub_pii(text: str) -> tuple[str, Counter[str]]:
    """Replace PII spans with placeholders; return the new text and per-type hit counts."""
    hits: Counter[str] = Counter()
    for name, pattern, replacement in PII_PATTERNS:
        text, n = pattern.subn(replacement, text)
        if n:
            hits[name] += n
    return text, hits


def clean_text(text: str) -> tuple[str, Counter[str]]:
    return scrub_pii(normalize_whitespace(text))
