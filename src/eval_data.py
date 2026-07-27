"""Load locked decision records + risk scores for evaluation experiments.

Does not change labels, features, or the risk-model specification.
Caches scored rows to results/locked_decisions.parquet so subsequent
experiments reuse the same predictions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_prep import build_decision_table
from .risk_model import (
    DEFAULT_TRAIN_FRAC,
    PRIMARY_TARGET,
    SEED,
    _train_one_target,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = REPO_ROOT / "results" / "locked_decisions.parquet"
VAL_FRAC_OF_TRAIN = 0.15  # final 15% of time-ordered train = validation


def _attach_drift_flags(
    df: pd.DataFrame,
    *,
    ref_scores: np.ndarray,
    batch_size: int = 100,
    psi_threshold: float = 0.2,
    bins: int = 10,
) -> pd.DataFrame:
    """Non-overlapping batch PSI of risk_score vs train reference."""
    from .monitors import population_stability_index

    out = df.copy()
    flags = np.zeros(len(out), dtype=bool)
    live: list[float] = []
    last_flag = False
    for i, score in enumerate(out["risk_score"].to_numpy(dtype=float)):
        live.append(float(score))
        if len(live) >= batch_size:
            last_flag = (
                population_stability_index(ref_scores, np.asarray(live), bins=bins)
                > psi_threshold
            )
            live = []
        flags[i] = last_flag
    out["drift_flag"] = flags.astype(int)
    return out


def build_locked_decisions(*, force_rebuild: bool = False) -> pd.DataFrame:
    """Return labelled decisions with locked primary risk_score + split flags.

    Split membership matches the locked risk model (70/30 time split).
    Validation = final 15% of the train period (thresholds set here only).
    """
    if CACHE_PATH.exists() and not force_rebuild:
        df = pd.read_parquet(CACHE_PATH)
        return df

    decisions = build_decision_table(verbose=False)
    # Primary model only — same seed / features / split as the locked run.
    primary = _train_one_target(
        decisions, PRIMARY_TARGET, train_frac=DEFAULT_TRAIN_FRAC, seed=SEED
    )
    df = primary["featured"].copy()
    assert df["in_train"].sum() == primary["metrics"]["n_train"]

    n_train = int(df["in_train"].sum())
    val_start = int(n_train * (1.0 - VAL_FRAC_OF_TRAIN))
    split = np.full(len(df), "test", dtype=object)
    split[:val_start] = "train"
    split[val_start:n_train] = "validation"
    df["split"] = split

    ref = df.loc[df["split"] == "train", "risk_score"].to_numpy(dtype=float)
    df = _attach_drift_flags(df, ref_scores=ref)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # parquet can't store tuples cleanly in some versions — drop list cols
    drop_cols = [c for c in ("seller_ids", "category_ids") if c in df.columns]
    df.drop(columns=drop_cols).to_parquet(CACHE_PATH, index=False)
    return df


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": df.loc[df["split"] == "train"].copy(),
        "validation": df.loc[df["split"] == "validation"].copy(),
        "test": df.loc[df["split"] == "test"].copy(),
    }
