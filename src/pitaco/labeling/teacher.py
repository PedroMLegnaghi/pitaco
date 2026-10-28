"""`pitaco label`: run a teacher LLM over a split and store predictions in the common format.

Default is a dry run that prints a cost estimate and calls nothing. Only `--execute` spends money.
"""

import asyncio
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field

from pitaco.config import BaseConfig
from pitaco.labeling.llm_client import (
    BudgetExceeded,
    LLMClient,
    ModelSpec,
    ResponseCache,
    estimate_tokens,
)
from pitaco.labeling.prompts import Mode, chat_messages, load_fewshot, teacher_system_prompt
from pitaco.predictions import PredictionRecord, record_from_raw, write_predictions
from pitaco.schema import strict_json_schema

SPLITS = ("gold_dev", "gold_test", "silver_train", "silver_val")
TYPICAL_OUTPUT_TOKENS = 120  # 1–3 aspects + summary; replaced by measured values after the dev run


class TeacherConfig(BaseConfig):
    prompts_dir: Path = Path("prompts")
    prompt_version: str = "v1"
    temperature: float = 0.0
    max_tokens: int = Field(400, gt=0)
    concurrency: int = Field(8, gt=0)
    timeout_s: float = 60.0
    response_format: Literal["json_schema", "json_object", "none"] = "json_schema"
    fuzzy_threshold: int = 90
    cache_path: Path = Path("data/cache/llm_responses.jsonl")
    models: dict[str, ModelSpec]


def system_name(model_key: str, mode: Mode, prompt_version: str) -> str:
    return f"teacher-{model_key}-{mode}-{prompt_version}"


def predictions_path(cfg: TeacherConfig, system: str, split: str) -> Path:
    return cfg.paths.results_dir / "predictions" / system / f"{split}.jsonl"


def load_split(cfg: TeacherConfig, split: str) -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return pd.read_json(cfg.paths.data_dir / "processed" / f"{split}.jsonl", lines=True)


def check_fewshot_not_in_gold(cfg: TeacherConfig) -> None:
    """Few-shot examples must come from silver; an example from gold would leak its label."""
    gold_ids = set()
    for split in ("gold_dev", "gold_test"):
        gold_ids |= set(load_split(cfg, split).review_id)
    leaked = {
        ex["review_id"] for ex in load_fewshot(cfg.prompts_dir, cfg.prompt_version)
    } & gold_ids
    if leaked:
        raise AssertionError(f"few-shot examples taken from gold: {sorted(leaked)}")


def response_format(cfg: TeacherConfig) -> dict | None:
    if cfg.response_format == "json_schema":
        schema = {"name": "review_analysis", "strict": True, "schema": strict_json_schema()}
        return {"type": "json_schema", "json_schema": schema}
    if cfg.response_format == "json_object":
        return {"type": "json_object"}
    return None


def estimate(cfg: TeacherConfig, model_key: str, mode: Mode, split: str, limit: int | None) -> dict:
    """Cost estimate from list prices and character counts. Calls nothing."""
    spec = cfg.models[model_key]
    reviews = load_split(cfg, split).head(limit) if limit else load_split(cfg, split)
    n = len(reviews)
    system_tokens = estimate_tokens(
        teacher_system_prompt(cfg.prompts_dir, cfg.prompt_version, mode)
    )
    review_tokens = sum(estimate_tokens(f"Avaliação: {t}") + 8 for t in reviews.text)
    output_typical = n * TYPICAL_OUTPUT_TOKENS * spec.output_per_m / 1e6
    uncached = (n * system_tokens + review_tokens) * spec.input_per_m / 1e6 + output_typical
    cached = (
        (system_tokens + review_tokens) * spec.input_per_m
        + (n - 1) * system_tokens * spec.cache_read_per_m
    ) / 1e6 + output_typical
    worst = (n * system_tokens + review_tokens) * spec.input_per_m / 1e6 + (
        n * cfg.max_tokens * spec.output_per_m / 1e6
    )
    return {
        "model": spec.id,
        "license": spec.license,
        "split": split,
        "n_reviews": n,
        "system_prompt_tokens": system_tokens,
        "usd_if_prefix_cached": round(cached, 4),
        "usd_if_not_cached": round(uncached, 4),
        "usd_worst_case": round(worst, 4),
    }


async def _label(
    cfg: TeacherConfig, spec: ModelSpec, mode: Mode, reviews: pd.DataFrame, system: str
) -> tuple[list[PredictionRecord], dict]:
    client = LLMClient(
        ResponseCache(cfg.cache_path),
        cfg.budget.max_usd,
        concurrency=cfg.concurrency,
        timeout_s=cfg.timeout_s,
    )
    spent_before = client.spent_usd
    system_prompt = teacher_system_prompt(cfg.prompts_dir, cfg.prompt_version, mode)
    fmt = response_format(cfg)
    stats = {
        "input_tokens": 0,
        "cached_input_tokens": 0,  # provider prefix-cache hits: decides the real cost
        "reasoning_tokens": 0,  # should stay 0 with reasoning disabled
        "from_cache": 0,
        "failed": 0,
        "budget_stopped": False,
    }

    async def one(row) -> PredictionRecord:
        try:
            resp = await client.complete(
                spec,
                chat_messages(system_prompt, row.text),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                response_format=fmt,
            )
        except BudgetExceeded:
            stats["budget_stopped"] = True
            raise
        except Exception as exc:  # network/provider errors: record and keep going
            stats["failed"] += 1
            return PredictionRecord(
                review_id=row.review_id,
                system=system,
                raw=None,
                analysis=None,
                valid_json=False,
                valid_schema=False,
                error=f"{type(exc).__name__}: {exc}"[:300],
            )
        stats["input_tokens"] += resp.input_tokens
        stats["cached_input_tokens"] += resp.cached_input_tokens
        stats["reasoning_tokens"] += resp.reasoning_tokens
        stats["from_cache"] += resp.from_cache
        return record_from_raw(
            row.review_id,
            system,
            resp.content,
            row.text,
            cfg.fuzzy_threshold,
            latency_s=resp.latency_s,
            cost_usd=resp.cost_usd,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
        )

    results = await asyncio.gather(*(one(r) for r in reviews.itertuples()), return_exceptions=True)
    records = [r for r in results if isinstance(r, PredictionRecord)]
    stats["spent_this_run_usd"] = round(client.spent_usd - spent_before, 4)
    stats["spent_total_usd"] = round(client.spent_usd, 4)
    return records, stats


def run(cfg: TeacherConfig, model_key: str, mode: Mode, split: str, limit: int | None) -> dict:
    check_fewshot_not_in_gold(cfg)
    reviews = load_split(cfg, split)
    if limit:
        reviews = reviews.head(limit)
    system = system_name(model_key, mode, cfg.prompt_version)
    records, stats = asyncio.run(_label(cfg, cfg.models[model_key], mode, reviews, system))
    write_predictions(predictions_path(cfg, system, split), records)

    n = len(records)
    n_evidence = sum(r.n_evidence for r in records)
    return {
        "system": system,
        "split": split,
        "n_written": n,
        "n_requested": len(reviews),
        "valid_schema_rate": round(sum(r.valid_schema for r in records) / max(n, 1), 4),
        "grounding_rate": round(sum(r.n_grounded for r in records) / max(n_evidence, 1), 4),
        **stats,
    }
