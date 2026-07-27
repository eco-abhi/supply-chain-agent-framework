"""exp2 — Threshold sensitivity: coverage/autonomy frontier on test."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from ..baselines import escalate_dynamic_router, quantiles_to_thresholds
from ..eval_data import build_locked_decisions, split_frames
from ..scoring import score_policy

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

# Exp1 operating point (must sit on the frontier)
EXP1_Q_COST = 0.80
EXP1_Q_RISK = 0.80


def run_exp2() -> pd.DataFrame:
    np.random.seed(SEED)
    df = build_locked_decisions()
    frames = split_frames(df)
    val, test = frames["validation"], frames["test"]
    cost_p95 = float(test["cost"].quantile(0.95))

    q_grid = np.round(np.linspace(0.50, 0.95, 10), 3)
    rows = []
    for qc in q_grid:
        for qr in q_grid:
            th = quantiles_to_thresholds(val, float(qc), float(qr))
            esc = escalate_dynamic_router(test, th)
            m = score_policy(test, esc, cost_p95=cost_p95)
            rows.append(
                {
                    "q_cost": float(qc),
                    "q_risk": float(qr),
                    "t_cost": th.t_cost,
                    "t_risk": th.t_risk,
                    "coverage": m["coverage"],
                    "autonomy": m["autonomy"],
                    "escalation_rate": m["escalation_rate"],
                    "latency_mean": m["latency_mean"],
                    "is_exp1_op": (
                        abs(qc - EXP1_Q_COST) < 1e-9 and abs(qr - EXP1_Q_RISK) < 1e-9
                    ),
                }
            )
    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "exp2.csv", index=False)

    # Pareto frontier: maximize coverage and autonomy
    pts = out[["coverage", "autonomy", "q_cost", "q_risk", "is_exp1_op"]].copy()
    frontier_mask = []
    for i, r in pts.iterrows():
        dominated = (
            (pts["coverage"] >= r["coverage"] - 1e-12)
            & (pts["autonomy"] >= r["autonomy"] - 1e-12)
            & (
                (pts["coverage"] > r["coverage"] + 1e-12)
                | (pts["autonomy"] > r["autonomy"] + 1e-12)
            )
        ).any()
        frontier_mask.append(not dominated)
    out["on_frontier"] = frontier_mask
    out.to_csv(RESULTS / "exp2.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(
        out["autonomy"],
        out["coverage"],
        c="#9aa7a0",
        s=28,
        label="grid",
        zorder=1,
    )
    fr = out.loc[out["on_frontier"]].sort_values("autonomy")
    ax.plot(
        fr["autonomy"],
        fr["coverage"],
        color="#1f5c4d",
        marker="o",
        label="frontier",
        zorder=2,
    )
    op = out.loc[out["is_exp1_op"]]
    if not op.empty:
        ax.scatter(
            op["autonomy"],
            op["coverage"],
            c="#b33a3a",
            s=80,
            marker="*",
            label="exp1 operating point",
            zorder=3,
        )
        on_f = bool(op["on_frontier"].iloc[0])
    else:
        on_f = False
    ax.set_xlabel("Autonomy P(auto | not high-stakes)")
    ax.set_ylabel("Coverage P(esc | high-stakes)")
    ax.set_title("Threshold sensitivity frontier (test)")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "frontier.png", dpi=150)
    plt.close(fig)

    print("=" * 72)
    print("EXP2 — Threshold sensitivity frontier")
    print("=" * 72)
    print(f"grid points: {len(out)}  frontier points: {int(out['on_frontier'].sum())}")
    print(f"exp1 operating point (q={EXP1_Q_COST}/{EXP1_Q_RISK}) on frontier: {on_f}")
    if not op.empty:
        print(
            f"  coverage={op['coverage'].iloc[0]:.4f}  "
            f"autonomy={op['autonomy'].iloc[0]:.4f}  "
            f"esc_rate={op['escalation_rate'].iloc[0]:.4f}"
        )
    print(f"wrote {RESULTS / 'exp2.csv'}")
    print(f"wrote {FIGURES / 'frontier.png'}")
    print("=" * 72)
    return out


if __name__ == "__main__":
    run_exp2()
