"""Governance policies / baselines for Olist evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scoring import DRIFT_RISK_BUMP, effective_risk


@dataclass
class Thresholds:
    """Absolute cutoffs derived from validation-set quantiles (never test)."""
    t_cost: float
    t_risk: float
    q_cost: float | None = None
    q_risk: float | None = None


def quantiles_to_thresholds(
    val: pd.DataFrame,
    q_cost: float,
    q_risk: float,
) -> Thresholds:
    """Map validation quantiles -> absolute T_cost / T_risk."""
    t_cost = float(val["cost"].quantile(q_cost))
    t_risk = float(val["risk_score"].quantile(q_risk))
    return Thresholds(t_cost=t_cost, t_risk=t_risk, q_cost=q_cost, q_risk=q_risk)


def escalate_static_centralized(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def escalate_static_decentralized(n: int) -> np.ndarray:
    return np.zeros(n, dtype=bool)


def escalate_fixed_hybrid(df: pd.DataFrame, t_cost: float) -> np.ndarray:
    """Cost-only cutoff."""
    return df["cost"].to_numpy(dtype=float) >= t_cost


def escalate_dynamic_router(
    df: pd.DataFrame,
    thresholds: Thresholds,
    *,
    drift_bump: float = DRIFT_RISK_BUMP,
) -> np.ndarray:
    """Escalate if cost >= T_cost OR effective_risk >= T_risk."""
    cost = df["cost"].to_numpy(dtype=float)
    risk = df["risk_score"].to_numpy(dtype=float)
    drift = df["drift_flag"].to_numpy(dtype=float)
    eff = effective_risk(risk, drift, bump=drift_bump)
    return (cost >= thresholds.t_cost) | (eff >= thresholds.t_risk)
