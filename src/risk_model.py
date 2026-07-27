"""LightGBM risk model: P(adverse | purchase-time features).

Primary target = late delivery OR cancel/unavailable.
Diagnostics: late-only and review-only models on the same feature set.
Time-based 70/30 split; no outcome leakage into features.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from .features import (
    RISK_FEATURE_COLS,
    add_risk_features,
    feature_matrix,
    fit_category_maps,
)

SEED = 42
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
DEFAULT_TRAIN_FRAC = 0.70

PRIMARY_TARGET = "adverse_outcome"
LATE_TARGET = "late_delivery"
REVIEW_TARGET = "adverse_review"


def _set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)


def time_split(
    df: pd.DataFrame,
    train_frac: float = DEFAULT_TRAIN_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-sorted decision frame into train / test."""
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    if cut <= 0 or cut >= len(df):
        raise ValueError(f"invalid train_frac={train_frac} for n={len(df)}")
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _impute_train_medians(
    train_X: pd.DataFrame, test_X: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    medians: dict[str, float] = {}
    train_out = train_X.copy()
    test_out = test_X.copy()
    for col in train_X.columns:
        # leave categorical codes alone (ints with -1 for unseen)
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


def _fit_lgbm(X_train: pd.DataFrame, y_train: np.ndarray, seed: int) -> lgb.LGBMClassifier:
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


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def _eval_split(
    model: lgb.LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, Any]:
    proba = model.predict_proba(X_test)[:, 1]
    auc = _safe_auc(y_test, proba)
    ap = float(average_precision_score(y_test, proba)) if y_test.sum() > 0 else float("nan")
    brier = float(brier_score_loss(y_test, proba))
    base = float(y_test.mean())
    importance = {
        f: float(v)
        for f, v in zip(X_test.columns, model.feature_importances_)
    }
    top20 = dict(
        sorted(importance.items(), key=lambda kv: -kv[1])[:20]
    )
    # calibration (quantile bins; fall back if too few positives)
    try:
        prob_true, prob_pred = calibration_curve(
            y_test, proba, n_bins=10, strategy="quantile"
        )
    except ValueError:
        prob_true, prob_pred = calibration_curve(
            y_test, proba, n_bins=5, strategy="uniform"
        )
    return {
        "test_auc": auc,
        "test_average_precision": ap,
        "test_brier": brier,
        "test_base_rate": base,
        "proba_test": proba,
        "y_test": y_test,
        "feature_importance": importance,
        "feature_importance_top20": top20,
        "calibration_prob_true": prob_true.tolist(),
        "calibration_prob_pred": prob_pred.tolist(),
    }


def _train_one_target(
    decisions: pd.DataFrame,
    target: str,
    *,
    train_frac: float,
    seed: int,
) -> dict[str, Any]:
    """Train on ``target`` using rates expanded from the same target."""
    featured = add_risk_features(decisions, outcome_col=target)
    train_df, test_df = time_split(featured, train_frac=train_frac)
    cat_maps = fit_category_maps(train_df)

    X_train_raw = feature_matrix(train_df, cat_maps)
    X_test_raw = feature_matrix(test_df, cat_maps)
    X_train, X_test, medians = _impute_train_medians(X_train_raw, X_test_raw)

    y_train = train_df[target].to_numpy(dtype=int)
    y_test = test_df[target].to_numpy(dtype=int)

    model = _fit_lgbm(X_train, y_train, seed)
    ev = _eval_split(model, X_test, y_test)

    # full-frame scores for downstream experiments (primary only used later)
    X_all = feature_matrix(featured, cat_maps)
    for col, med in medians.items():
        X_all[col] = X_all[col].fillna(med)
    featured = featured.copy()
    featured["risk_score"] = model.predict_proba(X_all)[:, 1]
    featured["in_train"] = np.arange(len(featured)) < len(train_df)

    return {
        "target": target,
        "model": model,
        "medians": medians,
        "cat_maps": cat_maps,
        "featured": featured,
        "train_df": train_df,
        "test_df": test_df,
        "metrics": {
            "target": target,
            "n_orders": int(len(featured)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "train_frac": train_frac,
            "base_rate_all": float(featured[target].mean()),
            "base_rate_train": float(y_train.mean()),
            "base_rate_test": float(y_test.mean()),
            "test_auc": ev["test_auc"],
            "test_average_precision": ev["test_average_precision"],
            "test_brier": ev["test_brier"],
            "calibration_prob_true": ev["calibration_prob_true"],
            "calibration_prob_pred": ev["calibration_prob_pred"],
            "feature_importance": ev["feature_importance"],
            "feature_importance_top20": ev["feature_importance_top20"],
            "feature_cols": list(RISK_FEATURE_COLS),
            "seed": seed,
            "train_end_timestamp": str(train_df["order_purchase_timestamp"].iloc[-1]),
            "test_start_timestamp": str(test_df["order_purchase_timestamp"].iloc[0]),
        },
        "proba_test": ev["proba_test"],
        "y_test": y_test,
    }


def _primary_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    m = bundle["metrics"]
    return m["primary"] if "primary" in m else m


def _plot_calibration(bundle: dict[str, Any], path: Path) -> None:
    m = _primary_metrics(bundle)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect")
    ax.plot(
        m["calibration_prob_pred"],
        m["calibration_prob_true"],
        marker="o",
        label="primary model",
    )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration — primary (late ∨ cancel)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_feature_importance(bundle: dict[str, Any], path: Path) -> None:
    top = _primary_metrics(bundle)["feature_importance_top20"]
    names = list(top.keys())[::-1]
    vals = list(top.values())[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(names, vals, color="#3d5a5b")
    ax.set_xlabel("LightGBM split importance")
    ax.set_title("Top feature importances — primary target")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_risk_model(
    decisions: pd.DataFrame,
    *,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    seed: int = SEED,
) -> dict[str, Any]:
    """Fit primary + late-only + review-only models; return primary bundle + all metrics."""
    _set_seeds(seed)

    primary = _train_one_target(
        decisions, PRIMARY_TARGET, train_frac=train_frac, seed=seed
    )
    late = _train_one_target(
        decisions, LATE_TARGET, train_frac=train_frac, seed=seed
    )
    review = _train_one_target(
        decisions, REVIEW_TARGET, train_frac=train_frac, seed=seed
    )

    metrics = {
        "primary": primary["metrics"],
        "late_only": late["metrics"],
        "review_only": review["metrics"],
        "summary": {
            "primary_auc": primary["metrics"]["test_auc"],
            "primary_ap": primary["metrics"]["test_average_precision"],
            "primary_brier": primary["metrics"]["test_brier"],
            "primary_base_rate_test": primary["metrics"]["base_rate_test"],
            "late_only_auc": late["metrics"]["test_auc"],
            "review_only_auc": review["metrics"]["test_auc"],
        },
        # flat keys for backward compatibility with earlier draft
        "n_orders": primary["metrics"]["n_orders"],
        "n_train": primary["metrics"]["n_train"],
        "n_test": primary["metrics"]["n_test"],
        "test_auc": primary["metrics"]["test_auc"],
        "test_average_precision": primary["metrics"]["test_average_precision"],
        "test_brier": primary["metrics"]["test_brier"],
        "adverse_base_rate_all": primary["metrics"]["base_rate_all"],
        "adverse_base_rate_test": primary["metrics"]["base_rate_test"],
        "feature_importance_top20": primary["metrics"]["feature_importance_top20"],
        "feature_cols": primary["metrics"]["feature_cols"],
        "seed": seed,
    }

    return {
        "model": primary["model"],
        "medians": primary["medians"],
        "cat_maps": primary["cat_maps"],
        "metrics": metrics,
        "featured": primary["featured"],
        "train_df": primary["train_df"],
        "test_df": primary["test_df"],
        "y_test": primary["y_test"],
        "proba_test": primary["proba_test"],
        "late_bundle": late,
        "review_bundle": review,
    }


def save_risk_metrics(
    metrics: dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    path = Path(path) if path else DEFAULT_RESULTS_DIR / "risk_model_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2)
    return path


def run_and_report(data_dir: Path | str | None = None) -> dict[str, Any]:
    """Build decisions, train three targets, print one-line summary, save artifacts."""
    from .data_prep import build_decision_table

    decisions = build_decision_table(data_dir)
    bundle = train_risk_model(decisions)
    metrics = bundle["metrics"]
    out = save_risk_metrics(metrics)

    figures = DEFAULT_FIGURES_DIR
    _plot_calibration(bundle, figures / "risk_calibration.png")
    _plot_feature_importance(bundle, figures / "risk_feature_importance.png")

    s = metrics["summary"]
    print(
        f"SUMMARY  primary AUC={s['primary_auc']:.4f} "
        f"AP={s['primary_ap']:.4f} "
        f"Brier={s['primary_brier']:.4f} "
        f"base={s['primary_base_rate_test']:.4f}  |  "
        f"late-only AUC={s['late_only_auc']:.4f}  |  "
        f"review-only AUC={s['review_only_auc']:.4f}"
    )
    print(
        f"n={metrics['n_orders']:,}  "
        f"train/test={metrics['n_train']:,}/{metrics['n_test']:,}  "
        f"wrote {out}  figures/risk_calibration.png  "
        f"figures/risk_feature_importance.png"
    )
    print("STOP — review before running exp1–exp5.")
    return bundle


if __name__ == "__main__":
    run_and_report()
