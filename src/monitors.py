"""PSI / disagreement monitors for drift and adversarial experiments."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def population_stability_index(
    reference: np.ndarray,
    live: np.ndarray,
    bins: int = 10,
) -> float:
    """Population Stability Index between two 1-d samples."""
    reference = np.asarray(reference, dtype=float)
    live = np.asarray(live, dtype=float)
    if reference.size == 0 or live.size == 0:
        return 0.0
    lo = float(min(reference.min(), live.min()))
    hi = float(max(reference.max(), live.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    ref_c, _ = np.histogram(reference, bins=edges)
    live_c, _ = np.histogram(live, bins=edges)
    ref_f = np.maximum(ref_c / max(ref_c.sum(), 1), 1e-4)
    live_f = np.maximum(live_c / max(live_c.sum(), 1), 1e-4)
    return float(np.sum((live_f - ref_f) * np.log(live_f / ref_f)))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_ztest(
    s1: int, n1: int, s2: int, n2: int
) -> tuple[float, float]:
    """Two-sided two-proportion z-test. Returns (z, p)."""
    if n1 <= 0 or n2 <= 0:
        return (float("nan"), float("nan"))
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    pval = float(2 * stats.norm.sf(abs(z)))
    return (float(z), pval)
