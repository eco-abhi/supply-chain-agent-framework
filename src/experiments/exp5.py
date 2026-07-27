"""exp5 — Retraining ablation: calendar vs PSI-triggered under randomized shifts."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import roc_auc_score

from ..data_prep import build_decision_table
from ..features import add_risk_features, feature_matrix, fit_category_maps
from ..monitors import population_stability_index

SEED = 42
N_SEEDS = 20
PSI_THRESHOLD = 0.2
CALENDAR_EVERY = 2000
PSI_BATCH = 500
WINDOW = 1500
AUC_STEP = 50          # fine grid — independent of retrain cadence
RECOVERY_TOL = 0.01
SYNTHETIC_SHIFT_SCALE = 1.75
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"


def _prepare_stream() -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = build_decision_table(verbose=False)
    featured = add_risk_features(decisions, outcome_col="adverse_outcome")
    cut = int(len(featured) * 0.70)
    train = featured.iloc[:cut].copy()
    test = featured.iloc[cut:].copy().reset_index(drop=True)
    return train, test


def _fit(train_df: pd.DataFrame, seed: int):
    cat_maps = fit_category_maps(train_df)
    X = feature_matrix(train_df, cat_maps)
    medians = {}
    for c in X.columns:
        if c.endswith("_code"):
            X[c] = X[c].fillna(-1)
            medians[c] = -1.0
        else:
            med = float(X[c].median()) if X[c].notna().any() else 0.0
            X[c] = X[c].fillna(med)
            medians[c] = med
    y = train_df["adverse_outcome"].to_numpy(dtype=int)
    model = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.08,
        num_leaves=23,
        random_state=seed,
        verbosity=-1,
        n_jobs=-1,
    )
    cats = [c for c in X.columns if c.endswith("_code")]
    model.fit(X, y, categorical_feature=cats if cats else "auto")
    return model, cat_maps, medians


def _predict_df(model, cat_maps, medians, df: pd.DataFrame) -> np.ndarray:
    X = feature_matrix(df, cat_maps)
    for c, m in medians.items():
        X[c] = X[c].fillna(m)
    return model.predict_proba(X)[:, 1]


def _rolling_auc(y: np.ndarray, p: np.ndarray, start: int, end: int) -> float:
    ys, ps = y[start:end], p[start:end]
    if len(ys) < 50 or len(np.unique(ys)) < 2:
        return float("nan")
    return float(roc_auc_score(ys, ps))


def _run_strategy(
    train0: pd.DataFrame,
    test: pd.DataFrame,
    *,
    strategy: str,
    seed: int,
    shift_at: int,
) -> dict:
    """Score the stream with calendar or PSI retrains; recover from shift_at."""
    test = test.copy()
    if "customer_seller_distance_km" in test.columns:
        test.loc[shift_at:, "customer_seller_distance_km"] *= SYNTHETIC_SHIFT_SCALE

    model, cat_maps, medians = _fit(train0, seed)

    n = len(test)
    proba = np.empty(n, dtype=float)
    y = test["adverse_outcome"].to_numpy(dtype=int)
    retrain_at: list[int] = []
    # Rolling prediction reference for PSI (recent scores), not frozen train —
    # detects abrupt post-shift jumps the way exp4's monitor does.
    ROLL_REF = 1000

    cursor = 0
    while cursor < n:
        if strategy == "calendar":
            next_retrain = ((cursor // CALENDAR_EVERY) + 1) * CALENDAR_EVERY
            end = min(next_retrain, n)
        else:
            end = min(cursor + PSI_BATCH, n)

        seg = test.iloc[cursor:end]
        proba[cursor:end] = _predict_df(model, cat_maps, medians, seg)

        do_retrain = False
        if strategy == "calendar" and end < n and end % CALENDAR_EVERY == 0:
            do_retrain = True
        elif strategy == "psi" and end < n and cursor >= ROLL_REF:
            ref = proba[cursor - ROLL_REF : cursor]
            live = proba[cursor:end]
            psi = population_stability_index(ref, live)
            if psi > PSI_THRESHOLD:
                do_retrain = True

        if do_retrain:
            pool = pd.concat([train0, test.iloc[:end]], axis=0)
            if len(pool) > 35000:
                pool = pool.iloc[-35000:]
            model, cat_maps, medians = _fit(pool, seed + end)
            retrain_at.append(end)

        cursor = end

    # Pre-shift baseline: rolling AUC on the window ending at shift_at
    baseline_auc = _rolling_auc(y, proba, max(0, shift_at - WINDOW), shift_at)
    if np.isnan(baseline_auc):
        baseline_auc = _rolling_auc(y, proba, 0, min(WINDOW, shift_at))

    # Fine-grained AUC path after shift (independent of retrain cadence).
    # Recovery requires a post-shift dip below the tolerance band, then return.
    auc_curve: list[tuple[int, float]] = []
    recovered_at: int | None = None
    dipped = False
    for t in range(shift_at + AUC_STEP, n + 1, AUC_STEP):
        a = _rolling_auc(y, proba, max(0, t - WINDOW), t)
        auc_curve.append((t, a))
        if np.isnan(a):
            continue
        if a < baseline_auc - RECOVERY_TOL:
            dipped = True
        if dipped and a >= baseline_auc - RECOVERY_TOL and recovered_at is None:
            recovered_at = t

    if not dipped:
        # Shift did not move rolling AUC outside the band
        decisions_to_recovery = 0
        recovery_end = shift_at
        recovered = True
        never_dipped = True
    elif recovered_at is None:
        decisions_to_recovery = n - shift_at
        recovery_end = n
        recovered = False
        never_dipped = False
    else:
        decisions_to_recovery = recovered_at - shift_at
        recovery_end = recovered_at
        recovered = True
        never_dipped = False

    # Under-governed high-risk during recovery: adverse with below pre-shift median score
    pre_med = float(np.median(proba[max(0, shift_at - WINDOW) : shift_at]))
    if recovery_end > shift_at:
        sl = slice(shift_at, recovery_end)
        under = int(((y[sl] == 1) & (proba[sl] < pre_med)).sum())
    else:
        under = 0

    # Curve aligned to decisions-since-shift for plotting
    curve_since_shift = [
        (t - shift_at, a) for t, a in auc_curve if t >= shift_at
    ]

    return {
        "strategy": strategy,
        "seed": seed,
        "shift_at": int(shift_at),
        "baseline_auc": float(baseline_auc) if baseline_auc == baseline_auc else float("nan"),
        "recovered": recovered,
        "never_dipped": never_dipped,
        "decisions_to_recovery": int(decisions_to_recovery),
        "under_governed_high_risk": under,
        "n_retrains": len(retrain_at),
        "n_retrains_after_shift": sum(1 for t in retrain_at if t >= shift_at),
        "auc_curve_since_shift": curve_since_shift,
    }


def run_exp5() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    print("exp5: preparing stream...")
    train0, test = _prepare_stream()
    n = len(test)
    # Leave room for pre-shift baseline window and post-shift recovery observation
    shift_lo = WINDOW + 500
    shift_hi = n - WINDOW - 500
    print(
        f"  train n={len(train0):,} base={train0['adverse_outcome'].mean():.4f}  "
        f"test n={n:,} base={test['adverse_outcome'].mean():.4f}"
    )
    print(
        f"  shift_at ~ Uniform[{shift_lo}, {shift_hi}]  "
        f"scale=×{SYNTHETIC_SHIFT_SCALE}  every seed"
    )

    records = []
    curves: dict[str, list] = {"calendar": [], "psi": []}
    for s in range(N_SEEDS):
        shift_at = int(rng.integers(shift_lo, shift_hi + 1))
        for strategy in ("calendar", "psi"):
            rec = _run_strategy(
                train0,
                test,
                strategy=strategy,
                seed=SEED + 17 * s + (0 if strategy == "calendar" else 1),
                shift_at=shift_at,
            )
            curves[strategy].append(rec["auc_curve_since_shift"])
            records.append({k: v for k, v in rec.items() if k != "auc_curve_since_shift"})
        print(f"  seed {s + 1}/{N_SEEDS} done  shift_at={shift_at}")

    out = pd.DataFrame(records)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "exp5.csv", index=False)

    # --- summary stats ---
    def _stats(s: pd.Series) -> dict:
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        return {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)),
            "iqr": q3 - q1,
            "q25": q1,
            "q75": q3,
        }

    print("=" * 72)
    print("EXP5 — Retraining ablation (20 seeds, randomized shift location)")
    print("=" * 72)
    print(f"calendar every {CALENDAR_EVERY} | PSI>{PSI_THRESHOLD} every {PSI_BATCH}")
    print(f"recovery = decisions from shift_at until rolling AUC ≥ baseline−{RECOVERY_TOL}")
    print(f"AUC grid step={AUC_STEP} (not snapped to retrain cadence)")
    print()

    summary = {}
    for strat in ("calendar", "psi"):
        sub = out.loc[out.strategy == strat]
        rec = _stats(sub["decisions_to_recovery"])
        ug = _stats(sub["under_governed_high_risk"])
        summary[strat] = {"recovery": rec, "under_governed": ug}
        print(f"  {strat}:")
        print(
            f"    decisions_to_recovery:  "
            f"mean={rec['mean']:.1f}  std={rec['std']:.1f}  "
            f"IQR=[{rec['q25']:.0f}, {rec['q75']:.0f}] (IQR={rec['iqr']:.1f})"
        )
        print(
            f"    under_governed:         "
            f"mean={ug['mean']:.1f}  std={ug['std']:.1f}  "
            f"IQR=[{ug['q25']:.0f}, {ug['q75']:.0f}] (IQR={ug['iqr']:.1f})"
        )
        print(
            f"    n_retrains:             "
            f"mean={sub['n_retrains'].mean():.1f}  "
            f"std={sub['n_retrains'].std(ddof=1):.1f}  "
            f"after_shift mean={sub['n_retrains_after_shift'].mean():.1f}"
        )

    cal = out.loc[out.strategy == "calendar"].sort_values("seed")
    psi = out.loc[out.strategy == "psi"].sort_values("seed")
    # pair by seed (same shift_at within seed)
    t_rec = ttest_rel(
        cal["decisions_to_recovery"].to_numpy(),
        psi["decisions_to_recovery"].to_numpy(),
    )
    t_ug = ttest_rel(
        cal["under_governed_high_risk"].to_numpy(),
        psi["under_governed_high_risk"].to_numpy(),
    )
    print()
    print(
        f"paired t recovery (calendar − psi): "
        f"t={t_rec.statistic:.3f}  p={t_rec.pvalue:.4g}  "
        f"meanΔ={cal['decisions_to_recovery'].mean()-psi['decisions_to_recovery'].mean():+.1f}"
    )
    print(
        f"paired t under-governed (calendar − psi): "
        f"t={t_ug.statistic:.3f}  p={t_ug.pvalue:.4g}  "
        f"meanΔ={cal['under_governed_high_risk'].mean()-psi['under_governed_high_risk'].mean():+.1f}"
    )

    # Variance comparison — the framework's actual prediction
    var_cal = float(cal["decisions_to_recovery"].var(ddof=1))
    var_psi = float(psi["decisions_to_recovery"].var(ddof=1))
    std_cal = float(cal["decisions_to_recovery"].std(ddof=1))
    std_psi = float(psi["decisions_to_recovery"].std(ddof=1))
    ratio = var_cal / var_psi if var_psi > 0 else float("inf")
    print()
    print("--- dip / recovery diagnostics ---")
    for strat in ("calendar", "psi"):
        sub = out.loc[out.strategy == strat]
        n_dip = int((~sub["never_dipped"]).sum()) if "never_dipped" in sub.columns else int((sub["decisions_to_recovery"] > 0).sum())
        print(f"  {strat}: dipped in {n_dip}/{len(sub)} seeds")
    print()
    print("--- variance comparison (decisions-to-recovery) ---")
    print(f"  calendar: var={var_cal:.1f}  std={std_cal:.1f}")
    print(f"  psi:      var={var_psi:.1f}  std={std_psi:.1f}")
    print(
        f"  calendar_var / psi_var = {ratio:.2f}  "
        f"→ calendar variance "
        f"{'>>' if ratio > 2 else '>' if ratio > 1 else '<='} trigger variance: "
        f"{ratio > 1}"
    )

    # Plot: per-seed curves with alpha + median ribbon
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"calendar": "#6b4f3a", "psi": "#1f5c4d"}
    for strat in ("calendar", "psi"):
        seed_curves = curves[strat]
        # interpolate each curve onto a common grid
        max_x = max((c[-1][0] for c in seed_curves if c), default=0)
        grid = np.arange(AUC_STEP, max_x + 1, AUC_STEP)
        mat = []
        for curve in seed_curves:
            if not curve:
                continue
            xs = np.array([p[0] for p in curve], dtype=float)
            ys = np.array([p[1] for p in curve], dtype=float)
            # forward-fill nan via interpolation ignoring nan
            valid = ~np.isnan(ys)
            if valid.sum() < 2:
                continue
            yi = np.interp(grid, xs[valid], ys[valid], left=np.nan, right=np.nan)
            mat.append(yi)
            ax.plot(grid, yi, color=colors[strat], alpha=0.18, lw=1)
        if mat:
            M = np.vstack(mat)
            med = np.nanmedian(M, axis=0)
            q25 = np.nanpercentile(M, 25, axis=0)
            q75 = np.nanpercentile(M, 75, axis=0)
            ax.plot(grid, med, color=colors[strat], lw=2.2, label=f"{strat} median")
            ax.fill_between(grid, q25, q75, color=colors[strat], alpha=0.2, label=f"{strat} IQR")
    ax.axhline(0, color="none")
    ax.set_xlabel("Decisions since shift onset")
    ax.set_ylabel(f"Rolling AUC (window={WINDOW})")
    ax.set_title("Recovery curves — per-seed spread (randomized shift location)")
    ax.legend(loc="lower right", fontsize=8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "recovery_curves.png", dpi=150)
    plt.close(fig)

    print(f"wrote {RESULTS / 'exp5.csv'}")
    print(f"wrote {FIGURES / 'recovery_curves.png'}")
    print("=" * 72)
    print("STOP — review exp4 + exp5.")
    return out


if __name__ == "__main__":
    run_exp5()
