"""Bootstrap confidence intervals over reviews.

Resample the test reviews with replacement many times and recompute the metric each time; the
spread of those values estimates how much the score would move on a different sample of 250
reviews. With n = 250 and rare aspects, differences of a few F1 points are often inside the noise.

The paired version resamples the *same* reviews for both systems, which removes the per-review
difficulty from the comparison and makes it much more sensitive than comparing two separate CIs.
"""

from collections.abc import Callable, Sequence

import numpy as np

Metric = Callable[[Sequence[str]], float]


def _resamples(ids: Sequence[str], n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    ids = list(ids)
    for _ in range(n_boot):
        yield [ids[i] for i in rng.integers(0, len(ids), size=len(ids))]


def bootstrap_ci(
    ids: Sequence[str], metric: Metric, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict:
    values = np.array([metric(sample) for sample in _resamples(ids, n_boot, seed)])
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return {"value": metric(ids), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap(
    ids: Sequence[str],
    metric_a: Metric,
    metric_b: Metric,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Difference A − B with CI; `p_b_not_worse` = share of resamples where B >= A."""
    diffs = np.array([metric_a(s) - metric_b(s) for s in _resamples(ids, n_boot, seed)])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "diff": metric_a(ids) - metric_b(ids),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_b_not_worse": float((diffs <= 0).mean()),
    }
