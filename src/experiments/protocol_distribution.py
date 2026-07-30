"""Section 5.4b — Protocol-selection characterization (descriptive only).

Deterministic complexity → {HTTP, MCP, A2A} on the locked TEST split.
No scoring and no latency model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from ..eval_data import build_locked_decisions, split_frames

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

PROTOCOL_ORDER = ("HTTP", "MCP", "A2A")


def assign_protocol(n_items: int, n_sellers: int) -> str | None:
    """Section 3.2 rule — exactly one bucket per order."""
    if n_items == 1 and n_sellers == 1:
        return "HTTP"
    if n_sellers == 1 and n_items > 1:
        return "MCP"
    if n_sellers > 1:
        return "A2A"
    return None


def run_protocol_distribution() -> pd.DataFrame:
    df = build_locked_decisions()
    test = split_frames(df)["test"].copy()
    assert len(test) == 29084, f"unexpected test size {len(test)}"

    n_items = test["n_items"].astype(int)
    n_sellers = test["n_distinct_sellers"].astype(int)
    n_categories = test["n_distinct_categories"].astype(int)

    assigned = [
        assign_protocol(int(i), int(s)) for i, s in zip(n_items, n_sellers)
    ]
    test["protocol_rule"] = assigned

    unmapped = test.loc[test["protocol_rule"].isna()]
    print(f"test orders: {len(test):,}")
    print(f"unmapped rows: {len(unmapped)} (expect 0)")
    if len(unmapped):
        print(unmapped[["order_id", "n_items", "n_distinct_sellers"]].head(20))
    assert len(unmapped) == 0, "protocol rule left orders unmapped"
    # every order in exactly one bucket
    assert set(test["protocol_rule"].unique()) <= set(PROTOCOL_ORDER)

    rows = []
    n_total = len(test)
    for proto in PROTOCOL_ORDER:
        sub = test.loc[test["protocol_rule"] == proto]
        n = len(sub)
        rows.append(
            {
                "protocol": proto,
                "n_orders": n,
                "pct_of_orders": n / n_total,
                "mean_n_items": float(sub["n_items"].mean()) if n else float("nan"),
                "mean_n_sellers": float(sub["n_distinct_sellers"].mean()) if n else float("nan"),
                "mean_n_categories": float(sub["n_distinct_categories"].mean())
                if n
                else float("nan"),
                "mean_cost": float(sub["cost"].mean()) if n else float("nan"),
                "mean_risk_score": float(sub["risk_score"].mean()) if n else float("nan"),
                "adverse_rate": float(sub["adverse_outcome"].mean()) if n else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "protocol_distribution.csv"
    out.to_csv(out_path, index=False)

    # Bar chart of shares
    fig, ax = plt.subplots(figsize=(5.5, 4))
    pcts = out["pct_of_orders"].to_numpy() * 100
    colors = {"HTTP": "#3d5a5b", "MCP": "#6b4f3a", "A2A": "#1f5c4d"}
    bars = ax.bar(
        out["protocol"],
        pcts,
        color=[colors[p] for p in out["protocol"]],
        width=0.65,
    )
    for bar, n, pct in zip(bars, out["n_orders"], pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{n:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("% of test orders")
    ax.set_xlabel("Protocol (complexity rule)")
    ax.set_title("Protocol distribution — Olist test split")
    ax.set_ylim(0, max(pcts.max() * 1.25, 10))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES / "protocol_distribution.png"
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # Writeup numbers
    print("=" * 72)
    print("Protocol distribution (test split, characterization only)")
    print("=" * 72)
    for _, r in out.iterrows():
        print(
            f"  {r['protocol']:4s}:  n={int(r['n_orders']):,}  "
            f"({100*r['pct_of_orders']:.2f}%)  "
            f"mean items={r['mean_n_items']:.2f}  "
            f"sellers={r['mean_n_sellers']:.2f}  "
            f"cats={r['mean_n_categories']:.2f}  "
            f"cost={r['mean_cost']:.1f}  "
            f"risk={r['mean_risk_score']:.3f}  "
            f"adverse={100*r['adverse_rate']:.2f}%"
        )
    shares = {r["protocol"]: r["pct_of_orders"] for _, r in out.iterrows()}
    degenerate = any(s < 0.005 or s > 0.995 for s in shares.values())
    print(
        f"non-degenerate three-way split: {not degenerate}  "
        f"(HTTP={100*shares['HTTP']:.2f}%, "
        f"MCP={100*shares['MCP']:.2f}%, "
        f"A2A={100*shares['A2A']:.2f}%)"
    )
    a2a_pct = 100 * shares["A2A"]
    a2a_n = int(out.loc[out.protocol == "A2A", "n_orders"].iloc[0])
    if a2a_pct < 1.0:
        a2a_note = (
            f"A2A (multi-seller) is a thin tail: {a2a_n:,} orders "
            f"({a2a_pct:.2f}% of test) — present but rare."
        )
    elif a2a_pct < 5.0:
        a2a_note = (
            f"A2A (multi-seller) is a small but non-trivial share: "
            f"{a2a_n:,} orders ({a2a_pct:.2f}% of test)."
        )
    else:
        a2a_note = (
            f"A2A (multi-seller) is material: {a2a_n:,} orders "
            f"({a2a_pct:.2f}% of test)."
        )
    print(f"writeup note: {a2a_note}")
    print(f"wrote {out_path}")
    print(f"wrote {fig_path}")
    print("=" * 72)
    print("STOP — protocol characterization only.")
    return out


if __name__ == "__main__":
    run_protocol_distribution()
