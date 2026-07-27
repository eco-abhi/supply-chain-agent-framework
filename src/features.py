"""Leakage-free purchase-time features for the Olist risk model.

Historical rates use expanding (past-only) windows ordered by
order_purchase_timestamp, with Bayesian shrinkage toward the global mean.
Realized outcomes never enter the feature matrix — only update rates after
each row is scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Shrinkage strength toward global mean: rate = (n_adv + a*g) / (n + a)
SHRINKAGE_A = 20.0

# Numeric + already-encoded columns passed to LightGBM.
# Categoricals (customer_state, seller_state, payment_type) are encoded in
# feature_matrix via train-fitted category codes.
NUMERIC_FEATURE_COLS = [
    "promised_days",
    "seller_adverse_rate",       # MAX across sellers (shrunk)
    "seller_adverse_rate_mean",  # MEAN across sellers (shrunk)
    "seller_n_prior_orders",     # MIN prior count across sellers
    "category_adverse_rate",     # MAX across categories (shrunk)
    "category_adverse_rate_mean",
    "customer_seller_distance_km",
    "same_state",
    "freight_value",
    "freight_to_price_ratio",
    "product_weight_g",
    "product_volume_cm3",
    "item_count",
    "purchase_month",
    "payment_installments",
]

CATEGORICAL_FEATURE_COLS = [
    "customer_state",
    "seller_state",
    "payment_type",
]

RISK_FEATURE_COLS = NUMERIC_FEATURE_COLS + [
    f"{c}_code" for c in CATEGORICAL_FEATURE_COLS
]

# Columns that must never be used as (or to derive) decision-time features.
FORBIDDEN_POST_PURCHASE_COLS = frozenset(
    {
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "shipping_limit_date",
        "review_creation_date",
        "review_answer_timestamp",
        "review_score",
        "adverse_review",
        "late_delivery",
        "adverse_outcome",
    }
)


def haversine_km(
    lat1: np.ndarray | pd.Series,
    lng1: np.ndarray | pd.Series,
    lat2: np.ndarray | pd.Series,
    lng2: np.ndarray | pd.Series,
) -> np.ndarray:
    """Great-circle distance in km between two lat/lng arrays."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lng1 = np.radians(np.asarray(lng1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lng2 = np.radians(np.asarray(lng2, dtype=float))
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _shrunk_rate(n_adverse: float, n_prior: int, global_rate: float, a: float) -> float:
    return (n_adverse + a * global_rate) / (n_prior + a)


def _expanding_multi_entity_rates(
    entity_lists: list[tuple],
    y: np.ndarray,
    *,
    a: float = SHRINKAGE_A,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Past-only Bayesian-shrunk rates aggregated across entities per order.

    Returns (rate_max, rate_mean, n_prior_min) aligned with input order.
    Updates each entity with the order outcome *after* emitting features.
    """
    n = len(y)
    rate_max = np.empty(n, dtype=float)
    rate_mean = np.empty(n, dtype=float)
    n_prior_min = np.empty(n, dtype=float)

    sum_so_far: dict = {}
    count_so_far: dict = {}
    global_adv = 0.0
    global_n = 0

    for i in range(n):
        g = (global_adv / global_n) if global_n > 0 else 0.0
        ents = entity_lists[i]
        if not ents:
            ents = ("__missing__",)

        rates = []
        priors = []
        for e in ents:
            c = count_so_far.get(e, 0)
            s = sum_so_far.get(e, 0.0)
            rates.append(_shrunk_rate(s, c, g, a))
            priors.append(c)

        rate_max[i] = max(rates)
        rate_mean[i] = float(np.mean(rates))
        n_prior_min[i] = float(min(priors))

        # update AFTER emitting features for this row
        yi = float(y[i])
        for e in ents:
            sum_so_far[e] = sum_so_far.get(e, 0.0) + yi
            count_so_far[e] = count_so_far.get(e, 0) + 1
        global_adv += yi
        global_n += 1

    return rate_max, rate_mean, n_prior_min


def assert_no_post_purchase_leakage(
    feature_cols: list[str] | None = None,
    frame_columns: list[str] | None = None,
) -> None:
    """Unit check: feature names must not reference post-purchase fields."""
    cols = feature_cols if feature_cols is not None else list(RISK_FEATURE_COLS)
    forbidden_hits = [
        c for c in cols
        if c in FORBIDDEN_POST_PURCHASE_COLS
        or any(tok in c for tok in (
            "delivered", "approved_at", "shipping_limit", "review_score",
            "carrier_date", "adverse_outcome", "adverse_review", "late_delivery",
        ))
    ]
    if forbidden_hits:
        raise AssertionError(
            f"Post-purchase / outcome leakage in features: {forbidden_hits}"
        )
    # promised_days is derived from estimated - purchase (purchase-time OK);
    # estimated delivery date itself must not appear as a raw feature column.
    if frame_columns is not None:
        raw_forbidden = FORBIDDEN_POST_PURCHASE_COLS.intersection(frame_columns)
        # raw forbidden cols may exist on the decision frame for labeling;
        # they simply must not be in the model feature list (checked above).
        _ = raw_forbidden


def add_risk_features(
    decisions: pd.DataFrame,
    *,
    outcome_col: str = "adverse_outcome",
    shrinkage_a: float = SHRINKAGE_A,
) -> pd.DataFrame:
    """Append leakage-free risk features. Rates expand on ``outcome_col`` only."""
    if outcome_col not in ("adverse_outcome", "late_delivery", "adverse_review"):
        raise ValueError(f"unexpected outcome_col={outcome_col}")

    df = decisions.sort_values("order_purchase_timestamp").reset_index(drop=True).copy()
    y = df[outcome_col].to_numpy(dtype=float)

    seller_lists = [
        t if isinstance(t, tuple) else (t,)
        for t in df["seller_ids"].tolist()
    ]
    category_lists = [
        t if isinstance(t, tuple) else (t,)
        for t in df["category_ids"].tolist()
    ]

    s_max, s_mean, s_n = _expanding_multi_entity_rates(
        seller_lists, y, a=shrinkage_a
    )
    c_max, c_mean, _ = _expanding_multi_entity_rates(
        category_lists, y, a=shrinkage_a
    )
    df["seller_adverse_rate"] = s_max
    df["seller_adverse_rate_mean"] = s_mean
    df["seller_n_prior_orders"] = s_n
    df["category_adverse_rate"] = c_max
    df["category_adverse_rate_mean"] = c_mean

    df["customer_seller_distance_km"] = haversine_km(
        df["cust_lat"], df["cust_lng"], df["seller_lat"], df["seller_lng"]
    )
    df["same_state"] = (
        df["customer_state"].astype(str) == df["seller_state"].astype(str)
    ).astype(int)

    # Ensure required columns exist
    required = [
        "promised_days",
        "freight_value",
        "freight_to_price_ratio",
        "product_weight_g",
        "product_volume_cm3",
        "item_count",
        "purchase_month",
        "payment_installments",
        "customer_state",
        "seller_state",
        "payment_type",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing expected columns: {missing}")

    assert_no_post_purchase_leakage(
        feature_cols=list(RISK_FEATURE_COLS),
        frame_columns=list(df.columns),
    )
    return df


def fit_category_maps(train_df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Map categorical string levels to ints from the training split only."""
    maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_FEATURE_COLS:
        levels = sorted({str(v) for v in train_df[col].fillna("missing").tolist()})
        maps[col] = {lv: i for i, lv in enumerate(levels)}
    return maps


def encode_categoricals(
    df: pd.DataFrame,
    cat_maps: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Apply train-fitted category codes; unseen levels -> -1."""
    out = df.copy()
    for col in CATEGORICAL_FEATURE_COLS:
        mapping = cat_maps[col]
        out[f"{col}_code"] = (
            out[col].fillna("missing").astype(str).map(mapping).fillna(-1).astype(int)
        )
    return out


def feature_matrix(
    df: pd.DataFrame,
    cat_maps: dict[str, dict[str, int]] | None = None,
) -> pd.DataFrame:
    """Return the model design matrix (decision-time features only)."""
    if cat_maps is not None:
        df = encode_categoricals(df, cat_maps)
    missing = [c for c in RISK_FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"missing feature columns {missing}; "
            "call add_risk_features and encode_categoricals first"
        )
    assert_no_post_purchase_leakage(list(RISK_FEATURE_COLS))
    return df[RISK_FEATURE_COLS].copy()
