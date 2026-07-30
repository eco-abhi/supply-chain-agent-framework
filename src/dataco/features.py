"""DataCo decision-time features — past-only rates, no post-shipment leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_prep import FORBIDDEN_POST_SHIPMENT

SHRINKAGE_A = 20.0

NUMERIC_FEATURE_COLS = [
    "days_scheduled",
    "order_item_quantity",
    "order_month",
    "order_weekday",
    "category_adverse_rate",
    "region_adverse_rate",
    "category_n_prior",
    "region_n_prior",
]

CATEGORICAL_FEATURE_COLS = [
    "shipping_mode",
    "market",
    "order_region",
    "order_country",
    "customer_segment",
    "category_name",
]

RISK_FEATURE_COLS = NUMERIC_FEATURE_COLS + [
    f"{c}_code" for c in CATEGORICAL_FEATURE_COLS
]

# Tokens that must not appear in feature names (unit check).
FORBIDDEN_NAME_TOKENS = (
    "days_real",
    "delivery_status",
    "late_delivery_risk",
    "late_by_days",
    "adverse_outcome",
    "order_profit",
    "shipping date",
    "profit",
    "benefit",
)


def _shrunk_rate(n_adverse: float, n_prior: int, global_rate: float, a: float) -> float:
    return (n_adverse + a * global_rate) / (n_prior + a)


def _expanding_single_entity_rates(
    groups: np.ndarray,
    y: np.ndarray,
    *,
    a: float = SHRINKAGE_A,
) -> tuple[np.ndarray, np.ndarray]:
    """Past-only Bayesian-shrunk adverse rate + prior count per entity."""
    n = len(y)
    rates = np.empty(n, dtype=float)
    priors = np.empty(n, dtype=float)
    sum_so_far: dict = {}
    count_so_far: dict = {}
    global_adv = 0.0
    global_n = 0
    for i in range(n):
        g = (global_adv / global_n) if global_n > 0 else 0.0
        e = groups[i]
        c = count_so_far.get(e, 0)
        s = sum_so_far.get(e, 0.0)
        rates[i] = _shrunk_rate(s, c, g, a)
        priors[i] = float(c)
        yi = float(y[i])
        sum_so_far[e] = s + yi
        count_so_far[e] = c + 1
        global_adv += yi
        global_n += 1
    return rates, priors


def assert_no_post_shipment_leakage(
    feature_cols: list[str] | None = None,
    frame_columns: list[str] | None = None,
) -> None:
    cols = feature_cols if feature_cols is not None else list(RISK_FEATURE_COLS)
    hits = [
        c
        for c in cols
        if c in FORBIDDEN_POST_SHIPMENT
        or any(tok in c.lower().replace(" ", "_") for tok in FORBIDDEN_NAME_TOKENS)
    ]
    # days_scheduled is allowed (shipment *scheduled* is decision-time)
    hits = [c for c in hits if c != "days_scheduled" and "scheduled" not in c.lower()]
    if hits:
        raise AssertionError(f"Post-shipment leakage in features: {hits}")
    if frame_columns is not None:
        # Forbidden raw columns may exist for labeling; must not be in feature list.
        overlap = FORBIDDEN_POST_SHIPMENT.intersection(cols)
        if overlap:
            raise AssertionError(f"Forbidden columns in feature list: {overlap}")


def add_risk_features(decisions: pd.DataFrame) -> pd.DataFrame:
    df = decisions.sort_values("order_date").reset_index(drop=True).copy()
    y = df["adverse_outcome"].to_numpy(dtype=float)

    cat_rate, cat_n = _expanding_single_entity_rates(
        df["category_name"].astype(str).to_numpy(), y
    )
    reg_rate, reg_n = _expanding_single_entity_rates(
        df["order_region"].astype(str).to_numpy(), y
    )
    df["category_adverse_rate"] = cat_rate
    df["category_n_prior"] = cat_n
    df["region_adverse_rate"] = reg_rate
    df["region_n_prior"] = reg_n

    required = [
        "days_scheduled",
        "order_item_quantity",
        "order_month",
        "order_weekday",
        "shipping_mode",
        "market",
        "order_region",
        "order_country",
        "customer_segment",
        "category_name",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    assert_no_post_shipment_leakage(
        feature_cols=list(RISK_FEATURE_COLS),
        frame_columns=list(df.columns),
    )
    return df


def fit_category_maps(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_FEATURE_COLS:
        levels = sorted({str(v) for v in train_df[col].fillna("missing").tolist()})
        maps[col] = {lv: i for i, lv in enumerate(levels)}
    return maps


def encode_categoricals(
    df: pd.DataFrame, cat_maps: dict[str, dict[str, int]]
) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_FEATURE_COLS:
        mapping = cat_maps[col]
        out[f"{col}_code"] = (
            out[col].fillna("missing").astype(str).map(mapping).fillna(-1).astype(int)
        )
    return out


def feature_matrix(
    df: pd.DataFrame, cat_maps: dict[str, dict[str, int]] | None = None
) -> pd.DataFrame:
    if cat_maps is not None:
        df = encode_categoricals(df, cat_maps)
    missing = [c for c in RISK_FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"missing feature columns {missing}")
    assert_no_post_shipment_leakage(list(RISK_FEATURE_COLS))
    return df[RISK_FEATURE_COLS].copy()
