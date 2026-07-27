"""Build one decision record per Olist order.

Reframe: each ORDER is a governance decision — auto-fulfill (decentralized)
vs escalate to review (centralized). No parts / stock / reorder modeling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

REQUIRED_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]

# Primary ground truth (Methods): late delivery OR cancel/unavailable.
# review_score is NEVER part of the primary target — see adverse_review.
ADVERSE_STATUSES = frozenset({"canceled", "unavailable"})
# Terminal labeled statuses retained in the evaluation set.
LABELED_STATUSES = frozenset({"delivered", "canceled", "unavailable"})
# Still open at dataset cutoff — dropped from the labeled set.
IN_TRANSIT_STATUSES = frozenset(
    {"created", "approved", "invoiced", "processing", "shipped"}
)

# Protocol surface operationalization (paper Methods):
#   single item & single seller  -> HTTP
#   single seller, multi-item    -> MCP
#   multi-seller                 -> A2A
PROTOCOL_HTTP = "HTTP"
PROTOCOL_MCP = "MCP"
PROTOCOL_A2A = "A2A"


def _require_files(data_dir: Path) -> None:
    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Olist CSVs in "
            f"{data_dir}: {missing}. See data/README.md for download instructions."
        )


def load_raw_tables(data_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load all Olist tables used by the evaluation pipeline."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    _require_files(data_dir)

    orders = pd.read_csv(data_dir / "olist_orders_dataset.csv")
    items = pd.read_csv(data_dir / "olist_order_items_dataset.csv")
    products = pd.read_csv(data_dir / "olist_products_dataset.csv")
    sellers = pd.read_csv(data_dir / "olist_sellers_dataset.csv")
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")
    payments = pd.read_csv(data_dir / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(data_dir / "olist_order_reviews_dataset.csv")
    geo = pd.read_csv(data_dir / "olist_geolocation_dataset.csv")
    cat_tr = pd.read_csv(data_dir / "product_category_name_translation.csv")
    cat_tr.columns = [c.lstrip("\ufeff") for c in cat_tr.columns]

    ts_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in ts_cols:
        orders[c] = pd.to_datetime(orders[c], errors="coerce")

    return {
        "orders": orders,
        "items": items,
        "products": products,
        "sellers": sellers,
        "customers": customers,
        "payments": payments,
        "reviews": reviews,
        "geo": geo,
        "category_translation": cat_tr,
    }


def _zip_lat_lng(geo: pd.DataFrame) -> pd.DataFrame:
    """Mean lat/lng per zip-code prefix (Olist has many rows per zip)."""
    g = geo.rename(
        columns={
            "geolocation_zip_code_prefix": "zip_prefix",
            "geolocation_lat": "lat",
            "geolocation_lng": "lng",
        }
    )
    return g.groupby("zip_prefix", as_index=False)[["lat", "lng"]].mean()


def _protocol_surface(n_items: int, n_sellers: int) -> str:
    if n_sellers > 1:
        return PROTOCOL_A2A
    if n_items > 1:
        return PROTOCOL_MCP
    return PROTOCOL_HTTP


def _complexity_level(complexity: int) -> int:
    if complexity <= 3:
        return 1
    if complexity <= 5:
        return 2
    return 3


def _assign_primary_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Primary adverse = late OR cancel/unavailable; drop in-transit; log edge cases."""
    stats: dict[str, int] = {}

    in_transit = df["order_status"].isin(IN_TRANSIT_STATUSES)
    stats["dropped_in_transit"] = int(in_transit.sum())
    df = df.loc[~in_transit].copy()

    # Unexpected statuses (if any) also dropped from the labeled set
    keep = df["order_status"].isin(LABELED_STATUSES)
    stats["dropped_other_status"] = int((~keep).sum())
    df = df.loc[keep].copy()

    delivered = df["order_delivered_customer_date"]
    estimated = df["order_estimated_delivery_date"]
    status = df["order_status"]

    bad_status = status.isin(ADVERSE_STATUSES)
    late = (
        delivered.notna()
        & estimated.notna()
        & (delivered > estimated)
    )

    # status==delivered but missing delivery timestamp -> non-adverse (not late)
    delivered_missing_ts = (status == "delivered") & delivered.isna()
    stats["delivered_missing_timestamp_non_adverse"] = int(delivered_missing_ts.sum())

    df["late_delivery"] = late.astype(int)
    # PRIMARY target — reviews excluded
    df["adverse_outcome"] = (late | bad_status).astype(int)

    # Diagnostic-only label (never used as primary target or as a feature)
    df["adverse_review"] = (
        df["review_score"].notna() & (df["review_score"] <= 2)
    ).astype(int)

    return df, stats


def build_decision_table(
    data_dir: Path | str | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return one labeled row per order with cost, complexity, protocol.

    Does NOT include risk_score (that comes from the risk model).
    Attaches item-level seller/category lists for multi-party rate features.
    """
    tables = load_raw_tables(data_dir)
    orders = tables["orders"]
    items = tables["items"]
    products = tables["products"]
    sellers = tables["sellers"]
    customers = tables["customers"]
    payments = tables["payments"]
    reviews = tables["reviews"]
    cat_tr = tables["category_translation"]
    zip_coords = _zip_lat_lng(tables["geo"])

    products = products.merge(cat_tr, on="product_category_name", how="left")
    products["category"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    ).fillna("unknown")
    products["product_volume_cm3"] = (
        products["product_length_cm"].fillna(0)
        * products["product_height_cm"].fillna(0)
        * products["product_width_cm"].fillna(0)
    )

    item_enriched = items.merge(
        products[
            [
                "product_id",
                "category",
                "product_weight_g",
                "product_volume_cm3",
            ]
        ],
        on="product_id",
        how="left",
    ).merge(
        sellers[["seller_id", "seller_zip_code_prefix", "seller_state"]],
        on="seller_id",
        how="left",
    )

    # Distinct sellers / categories per order (lists for multi-party rates)
    seller_lists = (
        item_enriched.groupby("order_id")["seller_id"]
        .agg(lambda s: tuple(dict.fromkeys(s.tolist())))
        .rename("seller_ids")
    )
    category_lists = (
        item_enriched.groupby("order_id")["category"]
        .agg(lambda s: tuple(dict.fromkeys(s.tolist())))
        .rename("category_ids")
    )

    agg = (
        item_enriched.groupby("order_id")
        .agg(
            price_total=("price", "sum"),
            freight_value=("freight_value", "sum"),
            item_count=("order_item_id", "count"),
            n_distinct_sellers=("seller_id", "nunique"),
            n_distinct_categories=("category", "nunique"),
            seller_id=("seller_id", "first"),
            category=("category", "first"),
            product_weight_g=("product_weight_g", "sum"),
            product_volume_cm3=("product_volume_cm3", "sum"),
            seller_zip_code_prefix=("seller_zip_code_prefix", "first"),
            seller_state=("seller_state", "first"),
        )
        .reset_index()
    )
    agg = agg.merge(seller_lists, on="order_id", how="left")
    agg = agg.merge(category_lists, on="order_id", how="left")

    agg["cost"] = agg["price_total"] + agg["freight_value"]
    agg["freight_to_price_ratio"] = np.where(
        agg["price_total"] > 0,
        agg["freight_value"] / agg["price_total"],
        np.nan,
    )
    agg["n_items"] = agg["item_count"]
    agg["complexity"] = (
        agg["item_count"] + agg["n_distinct_sellers"] + agg["n_distinct_categories"]
    )
    agg["complexity_level"] = agg["complexity"].map(_complexity_level)
    agg["protocol"] = [
        _protocol_surface(int(ni), int(ns))
        for ni, ns in zip(agg["item_count"], agg["n_distinct_sellers"])
    ]

    # Primary payment row: highest payment_value (installments + type)
    pay_sorted = payments.sort_values(
        ["order_id", "payment_value"], ascending=[True, False]
    )
    pay = (
        pay_sorted.groupby("order_id", as_index=False)
        .first()[["order_id", "payment_installments", "payment_type"]]
    )

    rev = (
        reviews.groupby("order_id", as_index=False)["review_score"]
        .min()
    )

    df = orders.merge(agg, on="order_id", how="inner")
    df = df.merge(customers, on="customer_id", how="left")
    df = df.merge(pay, on="order_id", how="left")
    df = df.merge(rev, on="order_id", how="left")

    df = df.merge(
        zip_coords.rename(
            columns={
                "zip_prefix": "customer_zip_code_prefix",
                "lat": "cust_lat",
                "lng": "cust_lng",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
    )
    df = df.merge(
        zip_coords.rename(
            columns={
                "zip_prefix": "seller_zip_code_prefix",
                "lat": "seller_lat",
                "lng": "seller_lng",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
    )

    # Purchase-time promise horizon (estimated delivery is known at order time)
    df["promised_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df, label_stats = _assign_primary_labels(df)

    df = df.dropna(subset=["order_purchase_timestamp"]).reset_index(drop=True)
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    df["cost_pctile"] = df["cost"].rank(pct=True, method="average")
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year

    if verbose:
        print(
            "label stats: "
            f"dropped_in_transit={label_stats['dropped_in_transit']}, "
            f"dropped_other_status={label_stats['dropped_other_status']}, "
            f"delivered_missing_ts_non_adverse="
            f"{label_stats['delivered_missing_timestamp_non_adverse']}, "
            f"labeled_orders={len(df)}, "
            f"primary_base_rate={df['adverse_outcome'].mean():.4f}, "
            f"late_rate={df['late_delivery'].mean():.4f}, "
            f"review_rate={df['adverse_review'].mean():.4f}"
        )

    df.attrs["label_stats"] = label_stats
    return df


if __name__ == "__main__":
    decisions = build_decision_table()
    print(
        f"orders={len(decisions):,}  "
        f"primary={decisions['adverse_outcome'].mean():.4f}  "
        f"late={decisions['late_delivery'].mean():.4f}  "
        f"review={decisions['adverse_review'].mean():.4f}"
    )
