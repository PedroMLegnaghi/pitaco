"""`pitaco data`: raw Olist CSV → cleaned, deduplicated reviews → frozen gold/silver splits."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from pydantic import Field

from pitaco.config import BaseConfig, StrictModel
from pitaco.data.clean import clean_text
from pitaco.data.dedup import dedup_key
from pitaco.data.download import download_reviews, sha256
from pitaco.data.split import make_splits
from pitaco.utils.seed import set_seed


class SourceConfig(StrictModel):
    kaggle_dataset: str
    reviews_file: str


class FilterConfig(StrictModel):
    min_words: int = Field(ge=1)


class SplitConfig(StrictModel):
    gold_dev_per_star: int
    gold_test_per_star: int
    silver_per_star: int
    silver_val_fraction: float = Field(gt=0, lt=1)
    near_dup_threshold: int = Field(ge=50, le=100)


class DataConfig(BaseConfig):
    source: SourceConfig
    filter: FilterConfig
    split: SplitConfig


# Data minimization: order_id (joins to customer tables), titles and answer timestamps are dropped.
KEEP = ["review_id", "review_score", "text", "review_creation_date"]


def load_and_clean(raw_csv: Path, min_words: int) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(raw_csv)
    stats: dict = {"raw_rows": len(raw)}

    df = raw.dropna(subset=["review_comment_message"]).copy()
    cleaned = df.review_comment_message.map(clean_text)
    df["text"] = cleaned.str[0]
    pii = sum(cleaned.str[1], Counter())
    stats["with_text"] = len(df)
    stats["pii_replacements"] = dict(sorted(pii.items()))
    stats["reviews_with_pii"] = int(cleaned.str[1].map(bool).sum())

    df = df[df.text.str.split().str.len() >= min_words]
    stats[f"min_{min_words}_words"] = len(df)

    df["key"] = df.text.map(dedup_key)
    df = df[df.key != ""].sort_values("review_id").drop_duplicates("key", keep="first")
    stats["unique_texts"] = len(df)
    stats["stars_unique"] = {
        int(k): int(v) for k, v in df.review_score.value_counts().sort_index().items()
    }
    return df[[*KEEP, "key"]].reset_index(drop=True), stats


def run(cfg: DataConfig) -> dict:
    set_seed(cfg.seed)
    data_dir = cfg.paths.data_dir
    raw_csv = download_reviews(cfg.source.kaggle_dataset, cfg.source.reviews_file, data_dir / "raw")

    df, stats = load_and_clean(raw_csv, cfg.filter.min_words)
    interim = data_dir / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim / "reviews_clean.parquet", index=False)

    s = cfg.split
    splits = make_splits(
        df,
        per_star_dev=s.gold_dev_per_star,
        per_star_test=s.gold_test_per_star,
        silver_per_star=s.silver_per_star,
        val_fraction=s.silver_val_fraction,
        near_dup_threshold=s.near_dup_threshold,
        seed=cfg.seed,
    )

    processed, ids_dir = data_dir / "processed", data_dir / "splits"
    processed.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.as_dict().items():
        frame[KEEP].to_json(
            processed / f"{name}.jsonl", orient="records", lines=True, force_ascii=False
        )
        (ids_dir / f"{name}.txt").write_text("\n".join(sorted(frame.review_id)) + "\n")

    manifest = {
        "source": {
            "file": cfg.source.reviews_file,
            "sha256": sha256(raw_csv),
            "license": "CC BY-NC-SA 4.0",
        },
        "seed": cfg.seed,
        "stats": stats,
        "near_duplicates_removed_from_pool": splits.removed_near_duplicates,
        "splits": {name: len(frame) for name, frame in splits.as_dict().items()},
    }
    (ids_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest
