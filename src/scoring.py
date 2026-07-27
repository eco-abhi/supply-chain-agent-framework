"""Scoring definitions shared by all experiments."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

# Assumed relative cost proxy (NOT measured wall-clock time):
#   base A2A = 1.0; +2.5 orchestrator if centralized; +20 if human approval.
LATENCY_A2A = 1.0
LATENCY_ORCHESTRATOR = 2.5
LATENCY_HUMAN = 20.0

DRIFT_RISK_BUMP = 0.2


def high_stakes_mask(df: pd.DataFrame, cost_p95: float) -> np.ndarray:
    """high_stakes := adverse_outcome==1 OR cost >= 95th pctile (of test cost)."""
    return (
        (df["adverse_outcome"].to_numpy(dtype=int) == 1)
        | (df["cost"].to_numpy(dtype=float) >= cost_p95)
    )


def effective_risk(risk: np.ndarray, drift: np.ndarray, bump: float = DRIFT_RISK_BUMP) -> np.ndarray:
    return np.clip(risk.astype(float) + bump * drift.astype(float), 0.0, 1.0)


def latency_units(escalated: np.ndarray, human: np.ndarray | None = None) -> np.ndarray:
    """Relative latency proxy per decision."""
    esc = escalated.astype(bool)
    if human is None:
        human = np.zeros(len(esc), dtype=bool)
    hum = human.astype(bool)
    lat = np.full(len(esc), LATENCY_A2A, dtype=float)
    lat[esc] = LATENCY_A2A + LATENCY_ORCHESTRATOR
    lat[hum] = LATENCY_A2A + LATENCY_ORCHESTRATOR + LATENCY_HUMAN
    return lat


def score_policy(
    df: pd.DataFrame,
    escalated: np.ndarray,
    *,
    cost_p95: float,
    human: np.ndarray | None = None,
    label_col: str = "adverse_outcome",
) -> dict[str, Any]:
    """coverage / autonomy / escalation_rate / latency + classifier metrics."""
    esc = np.asarray(escalated, dtype=bool)
    hs = high_stakes_mask(df, cost_p95)
    y = df[label_col].to_numpy(dtype=int)

    coverage = float(esc[hs].mean()) if hs.any() else float("nan")
    autonomy = float((~esc)[~hs].mean()) if (~hs).any() else float("nan")
    escalation_rate = float(esc.mean())
    lat = latency_units(esc, human)

    # Classifier view: escalated predicts positive realized outcome
    acc = float(accuracy_score(y, esc.astype(int)))
    f1 = float(f1_score(y, esc.astype(int), zero_division=0))
    # AUC needs scores; use escalated as binary score (0/1)
    try:
        auc = float(roc_auc_score(y, esc.astype(float)))
    except ValueError:
        auc = float("nan")

    return {
        "n": int(len(df)),
        "n_high_stakes": int(hs.sum()),
        "coverage": coverage,
        "autonomy": autonomy,
        "escalation_rate": escalation_rate,
        "latency_mean": float(lat.mean()),
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "cost_p95": float(cost_p95),
    }
