"""exp1 — Main benchmark with rank-based matched curves + coverage decomposition."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..baselines import (
    escalate_dynamic_router,
    escalate_fixed_hybrid,
    escalate_static_centralized,
    escalate_static_decentralized,
    quantiles_to_thresholds,
)
from ..eval_data import build_locked_decisions, split_frames
from ..scoring import DRIFT_RISK_BUMP, score_policy

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

DEFAULT_Q_COST = 0.80
DEFAULT_Q_RISK = 0.80
ESC_GRID = np.round(np.linspace(0.02, 0.50, 25), 4)


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def score_fixed_hybrid(test: pd.DataFrame) -> np.ndarray:
    return test["cost"].to_numpy(dtype=float)


def score_dynamic_scaled(
    test: pd.DataFrame,
    t_cost: float,
    t_risk: float,
    *,
    drift_bump: float = DRIFT_RISK_BUMP,
) -> np.ndarray:
    """max(cost/T_cost_val, risk_eff/T_risk_val); drift folded into risk."""
    cost = test["cost"].to_numpy(dtype=float)
    risk = test["risk_score"].to_numpy(dtype=float)
    drift = test["drift_flag"].to_numpy(dtype=float)
    risk_eff = np.clip(risk + drift_bump * drift, 0.0, 1.0)
    return np.maximum(cost / max(t_cost, 1e-12), risk_eff / max(t_risk, 1e-12))


def score_dynamic_rank_blend(
    test: pd.DataFrame,
    *,
    drift_bump: float = DRIFT_RISK_BUMP,
) -> np.ndarray:
    """max(pct_rank_cost, pct_rank_risk_eff) within the evaluation frame."""
    cost = test["cost"].to_numpy(dtype=float)
    risk = test["risk_score"].to_numpy(dtype=float)
    drift = test["drift_flag"].to_numpy(dtype=float)
    risk_eff = np.clip(risk + drift_bump * drift, 0.0, 1.0)
    n = len(test)
    pct_cost = (pd.Series(cost).rank(method="average").to_numpy() / n)
    pct_risk = (pd.Series(risk_eff).rank(method="average").to_numpy() / n)
    return np.maximum(pct_cost, pct_risk)


def escalate_by_score(score: np.ndarray, rate: float) -> np.ndarray:
    """Escalate the top ``rate`` fraction by score (ties broken stably)."""
    n = len(score)
    k = int(round(rate * n))
    k = int(np.clip(k, 0, n))
    if k == 0:
        return np.zeros(n, dtype=bool)
    order = np.argsort(-score, kind="mergesort")
    esc = np.zeros(n, dtype=bool)
    esc[order[:k]] = True
    return esc


# ---------------------------------------------------------------------------
# Coverage decomposition
# ---------------------------------------------------------------------------

def _masks(test: pd.DataFrame, cost_p95: float) -> dict[str, np.ndarray]:
    adv = test["adverse_outcome"].to_numpy(dtype=int) == 1
    hi = test["cost"].to_numpy(dtype=float) >= cost_p95
    return {
        "all": adv | hi,
        "adverse": adv & ~hi,
        "cost": hi & ~adv,
    }


def coverage_on(esc: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return float("nan")
    return float(esc[mask].mean())


def curve_for_score(
    test: pd.DataFrame,
    score: np.ndarray,
    *,
    config: str,
    cost_p95: float,
    rates: np.ndarray = ESC_GRID,
) -> pd.DataFrame:
    masks = _masks(test, cost_p95)
    rows = []
    prev_all = -1.0
    for rate in rates:
        esc = escalate_by_score(score, float(rate))
        realized = float(esc.mean())
        cov_all = coverage_on(esc, masks["all"])
        cov_adv = coverage_on(esc, masks["adverse"])
        cov_cost = coverage_on(esc, masks["cost"])
        # monotone check helper
        mono_ok = cov_all + 1e-12 >= prev_all
        prev_all = cov_all
        rows.append(
            {
                "config": config,
                "target_escalation_rate": float(rate),
                "realized_escalation_rate": realized,
                "coverage_all": cov_all,
                "coverage_adverse": cov_adv,
                "coverage_cost": cov_cost,
                "autonomy": float((~esc)[~masks["all"]].mean())
                if (~masks["all"]).any()
                else float("nan"),
                "monotone_ok": mono_ok,
            }
        )
    return pd.DataFrame(rows)


def is_monotone(series: pd.Series, tol: float = 1e-9) -> bool:
    v = series.to_numpy(dtype=float)
    return bool(np.all(np.diff(v) >= -tol))


# ---------------------------------------------------------------------------
# Ranker metrics (AUC / AP) + bootstrap CI + DeLong
# ---------------------------------------------------------------------------

def bootstrap_auc_ap(
    y: np.ndarray,
    score: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    auc = float(roc_auc_score(y, score))
    ap = float(average_precision_score(y, score))
    aucs, aps = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], score[idx]))
        aps.append(average_precision_score(y[idx], score[idx]))
    return {
        "auc": auc,
        "ap": ap,
        "auc_ci_lo": float(np.percentile(aucs, 2.5)),
        "auc_ci_hi": float(np.percentile(aucs, 97.5)),
        "ap_ci_lo": float(np.percentile(aps, 2.5)),
        "ap_ci_hi": float(np.percentile(aps, 97.5)),
    }


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks for DeLong covariance (Sun & Xu / pROC style)."""
    order = np.argsort(x, kind="mergesort")
    ranked = np.empty_like(x, dtype=float)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j < n - 1 and x[order[j]] == x[order[j + 1]]:
            j += 1
        # ranks i..j (1-based) share midrank
        mid = 0.5 * (i + j) + 1
        for k in range(i, j + 1):
            ranked[order[k]] = mid
        i = j + 1
    return ranked


def _mann_whitney_structural(
    sp: np.ndarray, sn: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """V10 (per positive) and V01 (per negative) via searchsorted — O(n log n)."""
    sn_sorted = np.sort(sn)
    sp_sorted = np.sort(sp)
    n_neg = len(sn_sorted)
    n_pos = len(sp_sorted)
    # for each positive: (#neg < pos + 0.5 * #neg == pos) / n_neg
    left = np.searchsorted(sn_sorted, sp, side="left")
    right = np.searchsorted(sn_sorted, sp, side="right")
    v10 = (left + 0.5 * (right - left)) / n_neg
    # for each negative: (#pos > neg + 0.5 * #pos == neg) / n_pos
    left_p = np.searchsorted(sp_sorted, sn, side="left")
    right_p = np.searchsorted(sp_sorted, sn, side="right")
    # #pos > neg = n_pos - right_p; #pos == neg = right_p - left_p
    v01 = ((n_pos - right_p) + 0.5 * (right_p - left_p)) / n_pos
    return v10, v01


def delong_auc_diff_pvalue(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    """Two-sided DeLong test for AUC(s1) - AUC(s2)."""
    y = np.asarray(y, dtype=int)
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    pos = y == 1
    neg = ~pos
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos < 1 or n_neg < 1:
        return {
            "auc1": float("nan"),
            "auc2": float("nan"),
            "diff": float("nan"),
            "z": float("nan"),
            "p": float("nan"),
        }

    def structural(scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        ranks = _compute_midrank(scores)
        auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        v10, v01 = _mann_whitney_structural(scores[pos], scores[neg])
        return float(auc), v10, v01

    auc1, v10_1, v01_1 = structural(s1)
    auc2, v10_2, v01_2 = structural(s2)

    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    L = np.array([1.0, -1.0])
    var = (L @ s10 @ L) / n_pos + (L @ s01 @ L) / n_neg
    diff = auc1 - auc2
    if var <= 0:
        z = 0.0 if abs(diff) < 1e-15 else float(np.sign(diff) * np.inf)
    else:
        z = diff / np.sqrt(var)
    from scipy.stats import norm

    p = float(2 * norm.sf(abs(z)))
    return {"auc1": auc1, "auc2": auc2, "diff": float(diff), "z": float(z), "p": p}


def interp_at(curve: pd.DataFrame, col: str, rate: float) -> float:
    sub = curve.sort_values("realized_escalation_rate")
    x = sub["realized_escalation_rate"].to_numpy(dtype=float)
    y = sub[col].to_numpy(dtype=float)
    _, idx = np.unique(x, return_index=True)
    return float(np.interp(rate, x[idx], y[idx]))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_coverage_all(curves: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {
        "fixed_hybrid": "#6b4f3a",
        "dynamic_router": "#1f5c4d",
        "risk_only": "#2c3e6b",
    }
    for cfg, color in colors.items():
        sub = curves.loc[curves["config"] == cfg].sort_values("realized_escalation_rate")
        if sub.empty:
            continue
        ax.plot(
            sub["realized_escalation_rate"],
            sub["coverage_all"],
            marker="o",
            ms=3,
            label=cfg,
            color=color,
        )
    ax.set_xlabel("Escalation rate (test)")
    ax.set_ylabel("coverage_all  P(esc | adverse ∨ cost≥p95)")
    ax.set_title("Coverage vs escalation (rank-based, monotone)")
    ax.legend()
    ax.set_xlim(0, 0.52)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_coverage_adverse(curves: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {
        "fixed_hybrid": "#6b4f3a",
        "dynamic_router": "#1f5c4d",
        "risk_only": "#2c3e6b",
    }
    for cfg, color in colors.items():
        sub = curves.loc[curves["config"] == cfg].sort_values("realized_escalation_rate")
        if sub.empty:
            continue
        ax.plot(
            sub["realized_escalation_rate"],
            sub["coverage_adverse"],
            marker="o",
            ms=3,
            label=cfg,
            color=color,
        )
    ax.set_xlabel("Escalation rate (test)")
    ax.set_ylabel("coverage_adverse  P(esc | adverse ∧ cost<p95)")
    ax.set_title("Risk-relevant coverage vs escalation")
    ax.legend()
    ax.set_xlim(0, 0.52)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_three_panel(curves: pd.DataFrame, path: Path) -> None:
    """Section 5.3 figure: coverage_all | coverage_adverse | coverage_cost."""
    colors = {
        "fixed_hybrid": "#6b4f3a",
        "dynamic_router": "#1f5c4d",
        "risk_only": "#2c3e6b",
    }
    panels = [
        ("coverage_all", "coverage_all\nP(esc | adverse ∨ high-cost)"),
        ("coverage_adverse", "coverage_adverse\nP(esc | adverse ∧ not high-cost)"),
        ("coverage_cost", "coverage_cost\nP(esc | high-cost ∧ not adverse)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, (col, title) in zip(axes, panels):
        for cfg, color in colors.items():
            sub = curves.loc[curves["config"] == cfg].sort_values(
                "realized_escalation_rate"
            )
            if sub.empty:
                continue
            ax.plot(
                sub["realized_escalation_rate"],
                sub[col],
                marker="o",
                ms=2.5,
                label=cfg,
                color=color,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Escalation rate")
        ax.set_xlim(0, 0.52)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Coverage")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Governance coverage by high-stakes reason", y=1.02, fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Operating-point table (unchanged threshold semantics)
# ---------------------------------------------------------------------------

def _run_operating_point(
    frames: dict[str, pd.DataFrame],
    *,
    q_cost: float,
    q_risk: float,
) -> tuple[pd.DataFrame, dict]:
    val, test = frames["validation"], frames["test"]
    cost_p95 = float(test["cost"].quantile(0.95))
    th = quantiles_to_thresholds(val, q_cost, q_risk)

    # risk_only: escalate when risk_score >= T_risk (validation quantile)
    risk_only_esc = test["risk_score"].to_numpy(dtype=float) >= th.t_risk

    policies = {
        "static_centralized": escalate_static_centralized(len(test)),
        "static_decentralized": escalate_static_decentralized(len(test)),
        "fixed_hybrid": escalate_fixed_hybrid(test, th.t_cost),
        "dynamic_router": escalate_dynamic_router(test, th),
        "risk_only": risk_only_esc,
    }
    rows = []
    for name, esc in policies.items():
        m = score_policy(test, esc, cost_p95=cost_p95)
        m["config"] = name
        m["t_cost"] = th.t_cost
        m["t_risk"] = th.t_risk
        m["q_cost"] = q_cost
        m["q_risk"] = q_risk
        rows.append(m)
    return pd.DataFrame(rows), {
        "thresholds": th,
        "cost_p95": cost_p95,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_exp1() -> dict:
    np.random.seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = build_locked_decisions()
    frames = split_frames(df)
    val, test = frames["validation"], frames["test"]
    th = quantiles_to_thresholds(val, DEFAULT_Q_COST, DEFAULT_Q_RISK)
    cost_p95 = float(test["cost"].quantile(0.95))

    # Three-way: cost-only, risk-only (= locked LightGBM P(adverse)), both
    scores = {
        "fixed_hybrid": score_fixed_hybrid(test),
        "dynamic_scaled": score_dynamic_scaled(test, th.t_cost, th.t_risk),
        "dynamic_rank_blend": score_dynamic_rank_blend(test),
        "risk_only": test["risk_score"].to_numpy(dtype=float),
    }

    curve_parts = [
        curve_for_score(test, sc, config=name, cost_p95=cost_p95)
        for name, sc in scores.items()
    ]
    curves_all = pd.concat(curve_parts, ignore_index=True)

    mid = curves_all["target_escalation_rate"].between(0.10, 0.30)
    mean_adv = curves_all.loc[mid].groupby("config")["coverage_adverse"].mean()
    preferred_dyn = (
        "dynamic_rank_blend"
        if mean_adv.get("dynamic_rank_blend", -1) >= mean_adv.get("dynamic_scaled", -1)
        else "dynamic_scaled"
    )
    curves = curves_all.copy()
    curves.loc[curves["config"] == preferred_dyn, "config"] = "dynamic_router"
    other_dyn = (
        "dynamic_scaled" if preferred_dyn == "dynamic_rank_blend" else "dynamic_rank_blend"
    )
    curves = curves.loc[curves["config"] != other_dyn].copy()

    curves_path = RESULTS / "exp1_curves.csv"
    curves.to_csv(curves_path, index=False)
    old = RESULTS / "exp1_matched_escalation.csv"
    if old.exists():
        old.unlink()

    _plot_coverage_all(curves, FIGURES / "coverage_vs_escalation.png")
    _plot_coverage_adverse(curves, FIGURES / "coverage_adverse_vs_escalation.png")
    _plot_three_panel(curves, FIGURES / "coverage_three_panel.png")

    table, meta = _run_operating_point(
        frames,
        q_cost=DEFAULT_Q_COST,
        q_risk=DEFAULT_Q_RISK,
    )
    table.to_csv(RESULTS / "exp1.csv", index=False)

    y = test["adverse_outcome"].to_numpy(dtype=int)
    ranker_scores = {
        "fixed_hybrid": scores["fixed_hybrid"],
        "dynamic_router": scores[preferred_dyn],
        "risk_only": scores["risk_only"],
    }
    ranker_rows = []
    for name, sc in ranker_scores.items():
        m = bootstrap_auc_ap(y, sc)
        m["config"] = name
        ranker_rows.append(m)
    rankers = pd.DataFrame(ranker_rows)
    rankers.to_csv(RESULTS / "exp1_ranker_metrics.csv", index=False)

    delong = delong_auc_diff_pvalue(
        y, ranker_scores["dynamic_router"], ranker_scores["fixed_hybrid"]
    )

    dyn_c = curves.loc[curves["config"] == "dynamic_router"]
    hyb_c = curves.loc[curves["config"] == "fixed_hybrid"]
    deltas = {}
    for r in (0.10, 0.15, 0.20, 0.25, 0.30):
        d = interp_at(dyn_c, "coverage_adverse", r)
        h = interp_at(hyb_c, "coverage_adverse", r)
        deltas[r] = {"dynamic": d, "hybrid": h, "delta": d - h}

    # cost coverage of risk_only at 50% esc (thesis claim)
    risk_c = curves.loc[curves["config"] == "risk_only"]
    risk_cost_at_50 = interp_at(risk_c, "coverage_cost", 0.50)

    mono = {}
    for cfg in ("fixed_hybrid", "dynamic_router", "risk_only"):
        sub = curves.loc[curves["config"] == cfg].sort_values("realized_escalation_rate")
        mono[cfg] = {
            "coverage_all": is_monotone(sub["coverage_all"]),
            "coverage_adverse": is_monotone(sub["coverage_adverse"]),
            "coverage_cost": is_monotone(sub["coverage_cost"]),
        }

    print("=" * 72)
    print("EXP1 — three-way: cost-only / risk-only / dynamic router")
    print("=" * 72)
    print(f"preferred dynamic score: {preferred_dyn}")
    print(
        f"T_cost_val={th.t_cost:.2f} BRL  T_risk_val={th.t_risk:.4f}  "
        f"cost_p95_test={cost_p95:.2f}"
    )
    print(f"risk_only coverage_cost at ~50% esc: {risk_cost_at_50:.4f}")
    print()
    print("--- coverage_adverse Δ (dynamic − hybrid) ---")
    for r, v in deltas.items():
        print(
            f"  esc={r:.0%}: dynamic={v['dynamic']:.4f}  "
            f"hybrid={v['hybrid']:.4f}  delta={v['delta']:+.4f}"
        )
    print()
    print("--- AUC / AP as rankers of adverse_outcome (test) ---")
    for _, row in rankers.iterrows():
        print(
            f"  {row['config']:16s}  "
            f"AUC={row['auc']:.4f} [{row['auc_ci_lo']:.4f}, {row['auc_ci_hi']:.4f}]  "
            f"AP={row['ap']:.4f} [{row['ap_ci_lo']:.4f}, {row['ap_ci_hi']:.4f}]"
        )
    print(
        f"  DeLong dynamic−hybrid: ΔAUC={delong['diff']:+.4f}  "
        f"z={delong['z']:.3f}  p={delong['p']:.4g}"
    )
    print()
    print("--- monotone? ---")
    for cfg, m in mono.items():
        print(
            f"  {cfg:16s}  all={m['coverage_all']}  "
            f"adverse={m['coverage_adverse']}  cost={m['coverage_cost']}"
        )
    print()
    cols = [
        "config", "latency_mean", "coverage", "autonomy",
        "escalation_rate", "accuracy", "f1", "auc",
    ]
    print("--- operating-point table (val q=0.80/0.80) ---")
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(f"wrote {RESULTS / 'exp1.csv'}")
    print(f"wrote {curves_path}")
    print(f"wrote {RESULTS / 'exp1_ranker_metrics.csv'}")
    print(f"wrote {FIGURES / 'coverage_vs_escalation.png'}")
    print(f"wrote {FIGURES / 'coverage_adverse_vs_escalation.png'}")
    print(f"wrote {FIGURES / 'coverage_three_panel.png'}")
    print("=" * 72)
    return {
        "table": table,
        "curves": curves,
        "rankers": rankers,
        "delong": delong,
        "deltas": deltas,
        "mono": mono,
        "preferred_dyn": preferred_dyn,
        "meta": meta,
        "risk_cost_at_50": risk_cost_at_50,
    }


if __name__ == "__main__":
    run_exp1()
