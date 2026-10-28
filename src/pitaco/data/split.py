"""Stratified, leakage-safe splits.

Gold (hand-labeled): `per_star_dev` + `per_star_test` reviews per star rating. Stratifying by stars
guarantees enough negative reviews — with natural sampling, half the test set would be 5-star
"chegou antes do prazo" and rare aspects would have almost no support.

Silver (teacher-labeled): sampled from what is left after removing gold near-duplicates, also
balanced by star so training and test distributions match. The real-world distribution is skewed
positive; that gap is reported as a limitation.
"""

from dataclasses import dataclass

import pandas as pd

from pitaco.data.dedup import near_duplicate_mask

STARS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Splits:
    gold_dev: pd.DataFrame
    gold_test: pd.DataFrame
    silver_train: pd.DataFrame
    silver_val: pd.DataFrame
    removed_near_duplicates: int

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "gold_dev": self.gold_dev,
            "gold_test": self.gold_test,
            "silver_train": self.silver_train,
            "silver_val": self.silver_val,
        }


def _per_star_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    parts = []
    for star in STARS:
        group = df[df.review_score == star]
        if len(group) < n:
            raise ValueError(f"only {len(group)} reviews with {star} stars, need {n}")
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts)


def make_splits(
    df: pd.DataFrame,
    *,
    per_star_dev: int,
    per_star_test: int,
    silver_per_star: int,
    val_fraction: float,
    near_dup_threshold: int,
    seed: int,
) -> Splits:
    """`df` must have one row per unique review with columns review_id, text, key, review_score."""
    df = df.sort_values("review_id").reset_index(drop=True)  # sampling independent of file order

    gold = _per_star_sample(df, per_star_dev + per_star_test, seed)
    dev = gold.groupby("review_score", group_keys=False).head(per_star_dev)
    test = gold.drop(dev.index)

    pool = df.drop(gold.index)
    near_dup = near_duplicate_mask(gold.key.tolist(), pool.key.tolist(), near_dup_threshold)
    pool = pool[~near_dup]

    silver = _per_star_sample(pool, silver_per_star, seed)
    n_val_per_star = round(silver_per_star * val_fraction)
    val = silver.groupby("review_score", group_keys=False).head(n_val_per_star)
    train = silver.drop(val.index)

    splits = Splits(dev, test, train, val, int(near_dup.sum()))
    assert_no_leakage(splits, near_dup_threshold)
    return splits


def assert_no_leakage(splits: Splits, near_dup_threshold: int) -> None:
    """Fail loudly if any gold review (or a near-copy of it) reached a silver split."""
    gold = pd.concat([splits.gold_dev, splits.gold_test])
    silver = pd.concat([splits.silver_train, splits.silver_val])
    frames = splits.as_dict().values()
    all_ids = pd.concat([f.review_id for f in frames])
    if all_ids.duplicated().any():
        raise AssertionError("a review_id appears in more than one split")
    if set(gold.key) & set(silver.key):
        raise AssertionError("an identical normalized text appears in gold and silver")
    if near_duplicate_mask(gold.key.tolist(), silver.key.tolist(), near_dup_threshold).any():
        raise AssertionError("a near-duplicate of a gold review appears in silver")
