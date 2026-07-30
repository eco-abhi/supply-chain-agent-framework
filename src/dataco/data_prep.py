"""DataCo decision records — leakage-free adverse label + cost.

Independent of the Olist pipeline. One row per order-item (unique
Order Item Id). Label rebuilt from day fields / delivery status;
Late_delivery_risk is never used as label or feature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "dataco" / "DataCoSupplyChainDataset.csv"
)

# Post-shipment / outcome fields — must never enter the feature matrix.
FORBIDDEN_POST_SHIPMENT = frozenset(
    {
        "Days for shipping (real)",
        "Delivery Status",
        "Late_delivery_risk",
        "Order Profit Per Order",
        "Order Item Profit Ratio",
        "Benefit per order",
        "shipping date (DateOrders)",
        "adverse_outcome",
        "late_by_days",
    }
)


def load_dataco(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. See data/dataco/README.md for download instructions."
        )
    return pd.read_csv(path, encoding="latin-1")


def build_decision_table(path: Path | str | None = None) -> pd.DataFrame:
    """Return time-sorted decision rows with cost + adverse_outcome."""
    raw = load_dataco(path)

    df = pd.DataFrame(
        {
            "order_item_id": raw["Order Item Id"],
            "order_id": raw["Order Id"],
            "order_date": pd.to_datetime(
                raw["order date (DateOrders)"], errors="coerce"
            ),
            "cost": raw["Order Item Total"].astype(float),
            "days_scheduled": raw["Days for shipment (scheduled)"],
            "days_real": raw["Days for shipping (real)"],
            "delivery_status": raw["Delivery Status"],
            "shipping_mode": raw["Shipping Mode"],
            "market": raw["Market"],
            "order_region": raw["Order Region"],
            "order_country": raw["Order Country"],
            "customer_segment": raw["Customer Segment"],
            "category_name": raw["Category Name"],
            "order_item_quantity": raw["Order Item Quantity"],
            # retained only for leakage asserts / diagnostics — never features
            "late_delivery_risk_col": raw["Late_delivery_risk"],
            "order_profit": raw["Order Profit Per Order"],
        }
    )

    # Label: rebuild from day fields (+ canceled). Do NOT use Late_delivery_risk.
    late_by_days = df["days_real"] > df["days_scheduled"]
    canceled = df["delivery_status"].astype(str).str.strip() == "Shipping canceled"
    # Status-based late is nearly identical; days rebuild is the ground rule.
    df["late_by_days"] = late_by_days.astype(int)
    df["adverse_outcome"] = (late_by_days | canceled).astype(int)

    df = df.dropna(subset=["order_date"]).sort_values("order_date").reset_index(drop=True)
    df["order_month"] = df["order_date"].dt.month
    df["order_weekday"] = df["order_date"].dt.weekday  # Mon=0
    df["cost_pctile"] = df["cost"].rank(pct=True, method="average")
    return df


if __name__ == "__main__":
    d = build_decision_table()
    print(
        f"n={len(d):,}  adverse_base={d['adverse_outcome'].mean():.4f}  "
        f"date_range={d['order_date'].min()} → {d['order_date'].max()}"
    )
