"""DataCo spine replication — risk model + rankers + coverage_adverse.

Does NOT touch Olist pipelines or outputs. Mirrors Olist Exp1 methodology:
same LightGBM config, 70/30 time split, rank-blend dynamic score, bootstrap
CIs, DeLong, and coverage_adverse vs escalation sweep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

# Allow `python -m src.dataco.replicate` and direct script runs.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.dataco.data_prep import build_decision_table  # noqa: E402
from src.dataco.features import (  # noqa: E402
    RISK_FEATURE_COLS,
    add_risk_features,
    assert_no_post_shipment_leakage,
    feature_matrix,
    fit_category_maps,
)
from src.experiments.exp1 import (  # noqa: E402
    ESC_GRID,
    bootstrap_auc_ap,
    curve_for_score,
    delong_auc_diff_pvalue,
    interp_at,
)

SEED = 42
TRAIN_FRAC = 0.70
RESULTS = _REPO / "results"
FIGURES = _REPO / "figures"

# Locked Olist numbers for side-by-side (RESULTS.md / exp1_ranker_metrics.csv).
OLIST = {
    "base_rate": 0.0635,
    "risk_auc": 0.719,
    "cost_auc": 0.497,
    "cost_auc_ci": (0.482, 0.511),
    "dynamic_auc": 0.633,
    "dynamic_auc_ci": (0.620, 0.645),
    "risk_ranker_auc": 0.719,
    "risk_ranker_auc_ci": (0.709, 0.730),
    "delong_diff": 0.136,
    "delong_z": 20.0,
    "delong_p": 5e-89,
    "cov_delta": {
        0.10: 0.068,
        0.15: 0.085,
        0.20: 0.124,
        0.25: 0.147,
        0.30: 0.180,
    },
}


def _set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)


def time_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    df = df.sort_values("order_date").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    if cut <= 0 or cut >= len(df):
        raise ValueError(f"invalid train_frac={train_frac} for n={len(df)}")
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _impute_train_medians(train_X: pd.DataFrame, test_X: pd.DataFrame):
    medians: dict[str, float] = {}
    train_out = train_X.copy()
    test_out = test_X.copy()
    for col in train_X.columns:
        if col.endswith("_code"):
            train_out[col] = train_out[col].fillna(-1)
            test_out[col] = test_out[col].fillna(-1)
            medians[col] = -1.0
            continue
        med = float(train_X[col].median()) if train_X[col].notna().any() else 0.0
        if np.isnan(med):
            med = 0.0
        medians[col] = med
        train_out[col] = train_out[col].fillna(med)
        test_out[col] = test_out[col].fillna(med)
    return train_out, test_out, medians


def fit_lgbm(X_train: pd.DataFrame, y_train: np.ndarray, seed: int) -> lgb.LGBMClassifier:
    """Same LightGBM config as Olist ``src.risk_model._fit_lgbm``."""
    cat_cols = [c for c in X_train.columns if c.endswith("_code")]
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=40,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_cols if cat_cols else "auto",
    )
    return model


def score_cost_only(test: pd.DataFrame) -> np.ndarray:
    return test["cost"].to_numpy(dtype=float)


def score_risk_only(test: pd.DataFrame) -> np.ndarray:
    return test["risk_score"].to_numpy(dtype=float)


def score_dynamic_rank_blend(test: pd.DataFrame) -> np.ndarray:
    """max(pct_rank_cost, pct_rank_risk) — Olist rank-blend without drift."""
    cost = test["cost"].to_numpy(dtype=float)
    risk = test["risk_score"].to_numpy(dtype=float)
    n = len(test)
    pct_cost = pd.Series(cost).rank(method="average").to_numpy() / n
    pct_risk = pd.Series(risk).rank(method="average").to_numpy() / n
    return np.maximum(pct_cost, pct_risk)


def _plot_coverage_adverse(curves: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {
        "cost_only": "#6b4f3a",
        "dynamic": "#1f5c4d",
    }
    labels = {"cost_only": "cost-only", "dynamic": "dynamic"}
    for cfg, color in colors.items():
        sub = curves.loc[curves["config"] == cfg].sort_values("realized_escalation_rate")
        if sub.empty:
            continue
        ax.plot(
            sub["realized_escalation_rate"],
            sub["coverage_adverse"],
            marker="o",
            ms=3,
            label=labels[cfg],
            color=color,
        )
    ax.set_xlabel("Escalation rate (test)")
    ax.set_ylabel("coverage_adverse  P(esc | adverse ∧ cost<p95)")
    ax.set_title("DataCo: risk-relevant coverage vs escalation")
    ax.legend()
    ax.set_xlim(0, 0.52)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def _fmt_p(p: float) -> str:
    return "≈ 0 (underflow)" if p == 0 else f"{p:.4g}"


def write_results_md(path: Path, summary: dict) -> None:
    r = summary
    lines = [
        "# DataCo replication results (side-by-side with Olist)",
        "",
        "Spine-only replication on DataCo Smart Supply Chain. Same methodology as",
        "Olist Exp1 (70/30 time split, LightGBM config, rank-blend dynamic score,",
        "bootstrap CIs, DeLong, `coverage_adverse` sweep). No threshold-sensitivity,",
        "retraining, adversarial, or protocol-distribution runs (single-supplier).",
        "",
        f"- DataCo n={r['n']:,} item-rows; train={r['n_train']:,} / test={r['n_test']:,}",
        f"  (cut at {r['split_date']}).",
        "- Label: `adverse = (days_real > days_scheduled) OR Shipping canceled`",
        "  (`Late_delivery_risk` unused). Cost = Order Item Total.",
        "",
        "## Base rate & risk model",
        "",
        "| Metric | DataCo | Olist |",
        "|---|---:|---:|",
        f"| Test base rate | **{r['test_base_rate']:.4f}** | {OLIST['base_rate']:.4f} |",
        f"| Risk model AUC | **{r['risk_auc']:.3f}** | {OLIST['risk_auc']:.3f} |",
        f"| Risk model AP | {r['risk_ap']:.3f} | — |",
        f"| Risk model Brier | {r['risk_brier']:.4f} | — |",
        "",
        f"Note: DataCo adverse base rate ≈ **{r['test_base_rate']:.2f}** (near 0.5 as expected;",
        f"overall {r['overall_base_rate']:.2f}). Olist test base rate is ~6%.",
        "",
        "## Ranker AUC of `adverse_outcome` (test)",
        "",
        "| Config | DataCo AUC [95% CI] | Olist AUC [95% CI] |",
        "|---|---|---|",
        (
            f"| Cost-only | **{r['cost_auc']:.3f}** {_fmt_ci(r['cost_auc_lo'], r['cost_auc_hi'])} "
            f"| {OLIST['cost_auc']:.3f} {_fmt_ci(*OLIST['cost_auc_ci'])} |"
        ),
        (
            f"| Dynamic | **{r['dyn_auc']:.3f}** {_fmt_ci(r['dyn_auc_lo'], r['dyn_auc_hi'])} "
            f"| {OLIST['dynamic_auc']:.3f} {_fmt_ci(*OLIST['dynamic_auc_ci'])} |"
        ),
        (
            f"| Risk-only | **{r['risk_ranker_auc']:.3f}** "
            f"{_fmt_ci(r['risk_ranker_auc_lo'], r['risk_ranker_auc_hi'])} "
            f"| {OLIST['risk_ranker_auc']:.3f} {_fmt_ci(*OLIST['risk_ranker_auc_ci'])} |"
        ),
        "",
        "### DeLong (dynamic − cost-only)",
        "",
        "| | DataCo | Olist |",
        "|---|---|---|",
        (
            f"| ΔAUC | **{r['delong_diff']:+.4f}** | "
            f"{OLIST['delong_diff']:+.3f} |"
        ),
        f"| z | **{r['delong_z']:.3f}** | {OLIST['delong_z']:.1f} |",
        (
            f"| p | **{_fmt_p(r['delong_p'])}** | ≈ {OLIST['delong_p']:.0e} |"
        ),
        "",
        "## `coverage_adverse` Δ (dynamic − cost) at matched escalation",
        "",
        "| Escalation | DataCo Δ | Olist Δ |",
        "|---:|---:|---:|",
    ]
    for rate in (0.10, 0.15, 0.20, 0.25, 0.30):
        d = r["cov_delta"][rate]
        lines.append(
            f"| {int(rate * 100)}% | **{d:+.3f}** | +{OLIST['cov_delta'][rate]:.3f} |"
        )
    lines += [
        "",
        "### Absolute coverage_adverse (DataCo)",
        "",
        "| Escalation | Dynamic | Cost-only | Δ |",
        "|---:|---:|---:|---:|",
    ]
    for rate in (0.10, 0.15, 0.20, 0.25, 0.30):
        lines.append(
            f"| {int(rate * 100)}% | {r['cov_dyn'][rate]:.3f} | "
            f"{r['cov_cost'][rate]:.3f} | {r['cov_delta'][rate]:+.3f} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `results/dataco_replication.csv`",
        "- `figures/dataco_coverage_adverse.png`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    _set_seeds(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    decisions = build_decision_table()
    featured = add_risk_features(decisions)
    assert_no_post_shipment_leakage(list(RISK_FEATURE_COLS))

    train, test = time_split(featured)
    cat_maps = fit_category_maps(train)
    X_train = feature_matrix(train, cat_maps)
    X_test = feature_matrix(test, cat_maps)
    X_train, X_test, _ = _impute_train_medians(X_train, X_test)
    y_train = train["adverse_outcome"].to_numpy(dtype=int)
    y_test = test["adverse_outcome"].to_numpy(dtype=int)

    model = fit_lgbm(X_train, y_train, SEED)
    proba = model.predict_proba(X_test)[:, 1]
    test = test.copy()
    test["risk_score"] = proba

    risk_auc = float(roc_auc_score(y_test, proba))
    risk_ap = float(average_precision_score(y_test, proba))
    risk_brier = float(brier_score_loss(y_test, proba))
    test_base = float(y_test.mean())
    overall_base = float(featured["adverse_outcome"].mean())

    s_cost = score_cost_only(test)
    s_risk = score_risk_only(test)
    s_dyn = score_dynamic_rank_blend(test)

    m_cost = bootstrap_auc_ap(y_test, s_cost)
    m_risk = bootstrap_auc_ap(y_test, s_risk)
    m_dyn = bootstrap_auc_ap(y_test, s_dyn)
    delong = delong_auc_diff_pvalue(y_test, s_dyn, s_cost)

    cost_p95 = float(np.quantile(test["cost"].to_numpy(dtype=float), 0.95))
    # Reuse Olist curve_for_score; config names for CSV clarity.
    curves_cost = curve_for_score(
        test, s_cost, config="cost_only", cost_p95=cost_p95, rates=ESC_GRID
    )
    curves_dyn = curve_for_score(
        test, s_dyn, config="dynamic", cost_p95=cost_p95, rates=ESC_GRID
    )
    curves = pd.concat([curves_cost, curves_dyn], ignore_index=True)
    _plot_coverage_adverse(curves, FIGURES / "dataco_coverage_adverse.png")

    cov_dyn: dict[float, float] = {}
    cov_cost: dict[float, float] = {}
    cov_delta: dict[float, float] = {}
    for rate in (0.10, 0.15, 0.20, 0.25, 0.30):
        d = interp_at(curves_dyn, "coverage_adverse", rate)
        c = interp_at(curves_cost, "coverage_adverse", rate)
        cov_dyn[rate] = d
        cov_cost[rate] = c
        cov_delta[rate] = d - c

    rows = [
        {
            "metric": "n",
            "dataco": len(featured),
            "olist": 96945,
            "note": "labelled decision rows",
        },
        {
            "metric": "n_train",
            "dataco": len(train),
            "olist": 67861,
            "note": "",
        },
        {
            "metric": "n_test",
            "dataco": len(test),
            "olist": 29084,
            "note": "",
        },
        {
            "metric": "test_base_rate",
            "dataco": test_base,
            "olist": OLIST["base_rate"],
            "note": "DataCo near ~0.5; Olist sparse",
        },
        {
            "metric": "risk_auc",
            "dataco": risk_auc,
            "olist": OLIST["risk_auc"],
            "note": "",
        },
        {
            "metric": "risk_ap",
            "dataco": risk_ap,
            "olist": np.nan,
            "note": "",
        },
        {
            "metric": "risk_brier",
            "dataco": risk_brier,
            "olist": np.nan,
            "note": "",
        },
        {
            "metric": "ranker_cost_auc",
            "dataco": m_cost["auc"],
            "olist": OLIST["cost_auc"],
            "note": f"CI [{m_cost['auc_ci_lo']:.4f},{m_cost['auc_ci_hi']:.4f}]",
        },
        {
            "metric": "ranker_dynamic_auc",
            "dataco": m_dyn["auc"],
            "olist": OLIST["dynamic_auc"],
            "note": f"CI [{m_dyn['auc_ci_lo']:.4f},{m_dyn['auc_ci_hi']:.4f}]",
        },
        {
            "metric": "ranker_risk_auc",
            "dataco": m_risk["auc"],
            "olist": OLIST["risk_ranker_auc"],
            "note": f"CI [{m_risk['auc_ci_lo']:.4f},{m_risk['auc_ci_hi']:.4f}]",
        },
        {
            "metric": "delong_dyn_minus_cost_diff",
            "dataco": delong["diff"],
            "olist": OLIST["delong_diff"],
            "note": f"z={delong['z']:.4f} p={delong['p']:.4g}",
        },
        {
            "metric": "delong_z",
            "dataco": delong["z"],
            "olist": OLIST["delong_z"],
            "note": "",
        },
        {
            "metric": "delong_p",
            "dataco": delong["p"],
            "olist": OLIST["delong_p"],
            "note": "",
        },
    ]
    for rate in (0.10, 0.15, 0.20, 0.25, 0.30):
        rows.append(
            {
                "metric": f"coverage_adverse_delta_at_{int(rate * 100)}pct",
                "dataco": cov_delta[rate],
                "olist": OLIST["cov_delta"][rate],
                "note": (
                    f"dyn={cov_dyn[rate]:.4f} cost={cov_cost[rate]:.4f}"
                ),
            }
        )
        rows.append(
            {
                "metric": f"coverage_adverse_dynamic_at_{int(rate * 100)}pct",
                "dataco": cov_dyn[rate],
                "olist": np.nan,
                "note": "",
            }
        )
        rows.append(
            {
                "metric": f"coverage_adverse_cost_at_{int(rate * 100)}pct",
                "dataco": cov_cost[rate],
                "olist": np.nan,
                "note": "",
            }
        )

    out_csv = RESULTS / "dataco_replication.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Also stash curve points for reproducibility (optional companion; main CSV is summary).
    curves.to_csv(RESULTS / "dataco_coverage_curves.csv", index=False)

    summary = {
        "n": len(featured),
        "n_train": len(train),
        "n_test": len(test),
        "split_date": str(test["order_date"].iloc[0].date()),
        "overall_base_rate": overall_base,
        "test_base_rate": test_base,
        "risk_auc": risk_auc,
        "risk_ap": risk_ap,
        "risk_brier": risk_brier,
        "cost_auc": m_cost["auc"],
        "cost_auc_lo": m_cost["auc_ci_lo"],
        "cost_auc_hi": m_cost["auc_ci_hi"],
        "dyn_auc": m_dyn["auc"],
        "dyn_auc_lo": m_dyn["auc_ci_lo"],
        "dyn_auc_hi": m_dyn["auc_ci_hi"],
        "risk_ranker_auc": m_risk["auc"],
        "risk_ranker_auc_lo": m_risk["auc_ci_lo"],
        "risk_ranker_auc_hi": m_risk["auc_ci_hi"],
        "delong_diff": delong["diff"],
        "delong_z": delong["z"],
        "delong_p": delong["p"],
        "cov_dyn": cov_dyn,
        "cov_cost": cov_cost,
        "cov_delta": cov_delta,
        "feature_cols": list(RISK_FEATURE_COLS),
    }
    (RESULTS / "dataco_replication_summary.json").write_text(
        json.dumps(
            {
                **{k: v for k, v in summary.items() if k not in ("cov_dyn", "cov_cost", "cov_delta")},
                "cov_dyn": {str(k): v for k, v in cov_dyn.items()},
                "cov_cost": {str(k): v for k, v in cov_cost.items()},
                "cov_delta": {str(k): v for k, v in cov_delta.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_results_md(_REPO / "RESULTS_DATACO.md", summary)

    print("--- DataCo spine replication ---")
    print(f"n={len(featured):,} train={len(train):,} test={len(test):,}")
    print(
        f"base_rate test={test_base:.4f} overall={overall_base:.4f}  "
        f"risk AUC={risk_auc:.4f} AP={risk_ap:.4f} Brier={risk_brier:.4f}"
    )
    print(
        f"rankers: cost={m_cost['auc']:.4f} dyn={m_dyn['auc']:.4f} "
        f"risk={m_risk['auc']:.4f}"
    )
    print(
        f"DeLong dyn−cost: Δ={delong['diff']:+.4f} z={delong['z']:.3f} "
        f"p={delong['p']:.4g}"
    )
    for rate in (0.10, 0.15, 0.20, 0.25, 0.30):
        print(
            f"  cov_adv @{int(rate*100)}%: dyn={cov_dyn[rate]:.3f} "
            f"cost={cov_cost[rate]:.3f} Δ={cov_delta[rate]:+.3f}"
        )
    print(f"wrote {out_csv}")
    print(f"wrote {FIGURES / 'dataco_coverage_adverse.png'}")
    print(f"wrote {_REPO / 'RESULTS_DATACO.md'}")
    return summary


if __name__ == "__main__":
    run()
