"""Expected-cost comparison for Section 5.9 (Olist + DataCo).

Uses locked test-set labels and risk scores. Does not retrain or re-split.
Policies: cost_only (score=cost) and dynamic_router (exp1 rank-blend).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..eval_data import build_locked_decisions, split_frames
from .exp1 import score_dynamic_rank_blend as olist_dynamic_rank_blend

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

DATACO_TEST_CACHE = RESULTS / "dataco_test_scores.parquet"
# Locked DataCo risk AUC from RESULTS_DATACO / dataco_replication.csv
DATACO_LOCKED_AUC = 0.7344
DATACO_AUC_TOL = 5e-4

RHO_REPORT = np.array([1, 2, 5, 10, 20, 50, 100], dtype=float)
# Fine log-grid for plots
RHO_FINE = np.unique(
    np.concatenate(
        [
            RHO_REPORT,
            np.round(np.logspace(np.log10(0.5), np.log10(200), 80), 6),
        ]
    )
)


def _escalate_prefix_fn_fp(
    y: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FN(e), FP(e) for escalating top-k by score, k=0..N.

    Returns arrays of length N+1 (one per k), and the escalation rates e=k/N.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n = len(y)
    if n == 0:
        raise ValueError("empty evaluation set")
    order = np.argsort(-score, kind="mergesort")
    y_ord = y[order]
    cum_pos = np.concatenate([[0], np.cumsum(y_ord)])  # length N+1; cum_pos[k]=pos in top-k
    total_pos = float(y.sum())
    k = np.arange(n + 1, dtype=float)
    tp = cum_pos.astype(float)
    fn = (total_pos - tp) / n
    fp = (k - tp) / n
    e = k / n
    return fn, fp, e


def min_expected_cost(
    y: np.ndarray, score: np.ndarray, rho: float
) -> tuple[float, float]:
    """Return (Cstar, e_star) for C = rho*FN + FP."""
    fn, fp, e = _escalate_prefix_fn_fp(y, score)
    c = rho * fn + fp
    i = int(np.argmin(c))
    return float(c[i]), float(e[i])


def cstar_curve(y: np.ndarray, score: np.ndarray, rhos: np.ndarray) -> np.ndarray:
    fn, fp, _ = _escalate_prefix_fn_fp(y, score)
    out = np.empty(len(rhos), dtype=float)
    for i, rho in enumerate(rhos):
        out[i] = float(np.min(rho * fn + fp))
    return out


def score_cost_only(df: pd.DataFrame) -> np.ndarray:
    return df["cost"].to_numpy(dtype=float)


def score_dataco_dynamic(df: pd.DataFrame) -> np.ndarray:
    """Rank-blend without drift (DataCo has no drift_flag; matches replicate)."""
    cost = df["cost"].to_numpy(dtype=float)
    risk = df["risk_score"].to_numpy(dtype=float)
    n = len(df)
    pct_cost = pd.Series(cost).rank(method="average").to_numpy() / n
    pct_risk = pd.Series(risk).rank(method="average").to_numpy() / n
    return np.maximum(pct_cost, pct_risk)


def load_olist_test() -> pd.DataFrame:
    df = build_locked_decisions(force_rebuild=False)
    test = split_frames(df)["test"]
    need = ["cost", "risk_score", "adverse_outcome", "drift_flag"]
    missing = [c for c in need if c not in test.columns]
    if missing:
        raise KeyError(f"Olist locked test missing {missing}")
    return test.reset_index(drop=True)


def _build_dataco_test_scores() -> pd.DataFrame:
    """Reproduce locked DataCo test scores (seeded) and cache — no new split."""
    from ..dataco.data_prep import build_decision_table
    from ..dataco.features import add_risk_features, feature_matrix, fit_category_maps
    from ..dataco.replicate import (
        SEED,
        TRAIN_FRAC,
        _impute_train_medians,
        _set_seeds,
        fit_lgbm,
        time_split,
    )

    _set_seeds(SEED)
    featured = add_risk_features(build_decision_table())
    train, test = time_split(featured, TRAIN_FRAC)
    cat_maps = fit_category_maps(train)
    X_train = feature_matrix(train, cat_maps)
    X_test = feature_matrix(test, cat_maps)
    X_train, X_test, _ = _impute_train_medians(X_train, X_test)
    y_train = train["adverse_outcome"].to_numpy(dtype=int)
    y_test = test["adverse_outcome"].to_numpy(dtype=int)
    model = fit_lgbm(X_train, y_train, SEED)
    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    if abs(auc - DATACO_LOCKED_AUC) > DATACO_AUC_TOL:
        raise RuntimeError(
            f"DataCo risk AUC {auc:.6f} != locked {DATACO_LOCKED_AUC} "
            f"(tol={DATACO_AUC_TOL}); refusing to proceed with mismatched scores."
        )
    out = pd.DataFrame(
        {
            "cost": test["cost"].to_numpy(dtype=float),
            "risk_score": proba,
            "adverse_outcome": y_test,
        }
    )
    DATACO_TEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATACO_TEST_CACHE, index=False)
    return out


def load_dataco_test() -> pd.DataFrame:
    """Load cached DataCo test scores; build+cache once if missing."""
    if DATACO_TEST_CACHE.exists():
        out = pd.read_parquet(DATACO_TEST_CACHE)
        need = ["cost", "risk_score", "adverse_outcome"]
        missing = [c for c in need if c not in out.columns]
        if missing:
            raise KeyError(f"DataCo cache missing {missing}")
        auc = float(
            roc_auc_score(out["adverse_outcome"], out["risk_score"])
        )
        if abs(auc - DATACO_LOCKED_AUC) > DATACO_AUC_TOL:
            raise RuntimeError(
                f"Cached DataCo AUC {auc:.6f} != locked {DATACO_LOCKED_AUC}"
            )
        return out.reset_index(drop=True)
    return _build_dataco_test_scores()


def router_beats_cost_rho_range(
    c_cost: np.ndarray, c_router: np.ndarray, rhos: np.ndarray
) -> tuple[float | None, float | None]:
    """Inclusive [rho_lo, rho_hi] on the fine grid where router < cost."""
    mask = c_router < c_cost
    if not mask.any():
        return None, None
    return float(rhos[mask].min()), float(rhos[mask].max())


def _plot_expected_cost(
    rhos: np.ndarray,
    c_cost: np.ndarray,
    c_router: np.ndarray,
    *,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rhos, c_cost, color="#6b4f3a", label="cost_only", lw=2)
    ax.plot(rhos, c_router, color="#1f5c4d", label="dynamic_router", lw=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\rho$  (relative cost of missed adverse)")
    ax.set_ylabel(r"$C^\star(\rho)=\min_e[\rho\,\mathrm{FN}(e)+\mathrm{FP}(e)]$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_dataset(
    name: str,
    test: pd.DataFrame,
    score_dynamic_fn,
) -> dict:
    y = test["adverse_outcome"].to_numpy(dtype=int)
    s_cost = score_cost_only(test)
    s_dyn = score_dynamic_fn(test)

    cstar_cost_r = cstar_curve(y, s_cost, RHO_REPORT)
    cstar_dyn_r = cstar_curve(y, s_dyn, RHO_REPORT)
    cstar_cost_f = cstar_curve(y, s_cost, RHO_FINE)
    cstar_dyn_f = cstar_curve(y, s_dyn, RHO_FINE)

    # Detailed e* at report rhos
    rows = []
    for rho, cc, cr in zip(RHO_REPORT, cstar_cost_r, cstar_dyn_r):
        _, e_c = min_expected_cost(y, s_cost, float(rho))
        _, e_r = min_expected_cost(y, s_dyn, float(rho))
        red = (cc - cr) / cc if cc > 0 else float("nan")
        rows.append(
            {
                "dataset": name,
                "rho": float(rho),
                "Cstar_cost": float(cc),
                "Cstar_router": float(cr),
                "pct_reduction": float(red),
                "e_star_cost": e_c,
                "e_star_router": e_r,
            }
        )

    rho10_i = int(np.where(RHO_REPORT == 10)[0][0])
    pct10 = float(
        (cstar_cost_r[rho10_i] - cstar_dyn_r[rho10_i]) / cstar_cost_r[rho10_i]
    )
    lo, hi = router_beats_cost_rho_range(cstar_cost_f, cstar_dyn_f, RHO_FINE)
    # Also check report grid
    report_beats = RHO_REPORT[cstar_dyn_r < cstar_cost_r]

    return {
        "name": name,
        "rows": rows,
        "pct_reduction_rho10": pct10,
        "rho_router_better_lo": lo,
        "rho_router_better_hi": hi,
        "report_rhos_router_better": report_beats.tolist(),
        "rhos_fine": RHO_FINE,
        "cstar_cost_fine": cstar_cost_f,
        "cstar_router_fine": cstar_dyn_f,
        "n": len(test),
        "base_rate": float(y.mean()),
    }


def run() -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    olist = evaluate_dataset(
        "olist",
        load_olist_test(),
        olist_dynamic_rank_blend,
    )
    dataco = evaluate_dataset(
        "dataco",
        load_dataco_test(),
        score_dataco_dynamic,
    )

    _plot_expected_cost(
        olist["rhos_fine"],
        olist["cstar_cost_fine"],
        olist["cstar_router_fine"],
        title="Olist: min expected cost vs ρ",
        path=FIGURES / "expected_cost_olist.png",
    )
    _plot_expected_cost(
        dataco["rhos_fine"],
        dataco["cstar_cost_fine"],
        dataco["cstar_router_fine"],
        title="DataCo: min expected cost vs ρ",
        path=FIGURES / "expected_cost_dataco.png",
    )

    table = pd.DataFrame(olist["rows"] + dataco["rows"])
    out_csv = RESULTS / "expected_cost.csv"
    table.to_csv(out_csv, index=False)

    print("--- Expected-cost comparison (Section 5.9) ---")
    for res in (olist, dataco):
        print(
            f"\n[{res['name']}] n={res['n']:,}  base_rate={res['base_rate']:.4f}"
        )
        print(
            f"  ρ=10  % reduction (cost→router): "
            f"{100 * res['pct_reduction_rho10']:.2f}%"
        )
        lo, hi = res["rho_router_better_lo"], res["rho_router_better_hi"]
        print(
            f"  C*_router < C*_cost on ρ ∈ [{lo}, {hi}] "
            f"(fine grid); report-grid wins: {res['report_rhos_router_better']}"
        )
        sub = table.loc[table["dataset"] == res["name"]]
        for _, r in sub.iterrows():
            print(
                f"  ρ={r['rho']:>5.0f}  C*_cost={r['Cstar_cost']:.4f}  "
                f"C*_router={r['Cstar_router']:.4f}  "
                f"Δ%={100 * r['pct_reduction']:+.2f}%"
            )

    print(f"\nρ=10 reductions:  Olist={100 * olist['pct_reduction_rho10']:.2f}%  "
          f"DataCo={100 * dataco['pct_reduction_rho10']:.2f}%")
    print(f"wrote {out_csv}")
    print(f"wrote {FIGURES / 'expected_cost_olist.png'}")
    print(f"wrote {FIGURES / 'expected_cost_dataco.png'}")
    return table


if __name__ == "__main__":
    run()
