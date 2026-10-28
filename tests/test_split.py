import pandas as pd
import pytest

from pitaco.data.dedup import dedup_key, near_duplicate_mask
from pitaco.data.split import Splits, assert_no_leakage, make_splits

PARAMS = dict(
    per_star_dev=2, per_star_test=4, silver_per_star=10, val_fraction=0.2, near_dup_threshold=90
)


def _corpus(n_per_star: int = 40) -> pd.DataFrame:
    rows = []
    for star in range(1, 6):
        for i in range(n_per_star):
            text = f"avaliacao {star} numero {i} " + "abcdefghij"[i % 10] * (i + 3)
            rows.append({"review_id": f"r{star}_{i:03d}", "text": text, "review_score": star})
    df = pd.DataFrame(rows)
    df["key"] = df.text.map(dedup_key)
    return df


def test_dedup_key_normalizes_case_accents_punctuation():
    assert dedup_key("Ótimo!!  Recomendo.") == dedup_key("otimo recomendo") == "otimo recomendo"


def test_near_duplicate_mask():
    pool = ["produto chegou antes do prazo", "produto chegou antes do prazo!", "veio quebrado"]
    mask = near_duplicate_mask(["produto chegou antes do prazo"], [dedup_key(p) for p in pool], 90)
    assert mask.tolist() == [True, True, False]


def test_split_sizes_are_stratified_by_star():
    s = make_splits(_corpus(), seed=0, **PARAMS)
    for frame, per_star in [
        (s.gold_dev, 2),
        (s.gold_test, 4),
        (s.silver_train, 8),
        (s.silver_val, 2),
    ]:
        assert frame.review_score.value_counts().to_dict() == {
            star: per_star for star in range(1, 6)
        }


def test_splits_are_deterministic_and_order_independent():
    df = _corpus()
    a = make_splits(df, seed=0, **PARAMS)
    b = make_splits(df.sample(frac=1, random_state=123), seed=0, **PARAMS)
    for name in a.as_dict():
        assert set(a.as_dict()[name].review_id) == set(b.as_dict()[name].review_id)


def test_near_copy_of_gold_never_reaches_silver():
    df = _corpus()
    gold_text = make_splits(df, seed=0, **PARAMS).gold_test.text.iloc[0]
    copy = {"review_id": "zzz_copy", "text": gold_text + "!", "review_score": 1}
    df = pd.concat([df, pd.DataFrame([copy])], ignore_index=True)
    df["key"] = df.text.map(dedup_key)
    s = make_splits(df, seed=0, **PARAMS)
    silver_ids = set(s.silver_train.review_id) | set(s.silver_val.review_id)
    assert "zzz_copy" not in silver_ids
    assert s.removed_near_duplicates >= 1


def test_leakage_check_catches_overlap():
    s = make_splits(_corpus(), seed=0, **PARAMS)
    leaked = Splits(
        s.gold_dev, s.gold_test, pd.concat([s.silver_train, s.gold_test.head(1)]), s.silver_val, 0
    )
    with pytest.raises(AssertionError):
        assert_no_leakage(leaked, 90)


def test_not_enough_reviews_for_a_star_fails_loudly():
    with pytest.raises(ValueError, match="stars"):
        make_splits(_corpus(n_per_star=5), seed=0, **PARAMS)
