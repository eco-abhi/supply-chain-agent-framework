"""exp3 — Learned baseline vs threshold router (combined-job oracle).

Learned policies train on [cost, risk, complexity, drift] (+ optional cost*risk)
to predict the dual-axis oracle high_stakes := adverse OR cost >= val-p95.
Distinct from exp1 risk_only (raw P(adverse)).
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ..baselines import escalate_dynamic_router, quantiles_to_thresholds
from ..eval_data import build_locked_decisions, split_frames
from ..scoring import high_stakes_mask, score_policy

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
BASE_FEATURES = ["cost", "risk_score", "complexity", "drift_flag"]
Q_COST, Q_RISK = 0.80, 0.80


def _matrix(df: pd.DataFrame, *, with_interaction: bool) -> np.ndarray:
    X = df[BASE_FEATURES].to_numpy(dtype=float).copy()
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(med, inds[1])
    if with_interaction:
        inter = df["cost"].to_numpy(dtype=float) * df["risk_score"].to_numpy(dtype=float)
        inter = np.nan_to_num(inter, nan=float(np.nanmedian(inter)))
        X = np.column_stack([X, inter])
    return X


def _metrics(y: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "ap": float(average_precision_score(y, proba)) if y.sum() > 0 else float("nan"),
    }


def _best_threshold(y_va: np.ndarray, p_va: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        f1 = f1_score(y_va, (p_va >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def escalate_dynamic_with_interaction(
    df: pd.DataFrame, t_cost: float, t_risk: float, t_inter: float
) -> np.ndarray:
    cost = df["cost"].to_numpy(dtype=float)
    risk = df["risk_score"].to_numpy(dtype=float)
    drift = df["drift_flag"].to_numpy(dtype=float)
    eff = np.clip(risk + 0.2 * drift, 0.0, 1.0)
    return (cost >= t_cost) | (eff >= t_risk) | ((cost * risk) >= t_inter)


def run_exp3() -> pd.DataFrame:
    np.random.seed(SEED)
    df = build_locked_decisions()
    frames = split_frames(df)
    train, val, test = frames["train"], frames["validation"], frames["test"]

    cost_p95_val = float(val["cost"].quantile(0.95))
    cost_p95_test = float(test["cost"].quantile(0.95))
    y_tr = high_stakes_mask(train, cost_p95_val).astype(int)
    y_va = high_stakes_mask(val, cost_p95_val).astype(int)
    y_te = high_stakes_mask(test, cost_p95_val).astype(int)

    rows = []
    for with_inter in (False, True):
        X_tr = _matrix(train, with_interaction=with_inter)
        X_va = _matrix(val, with_interaction=with_inter)
        X_te = _matrix(test, with_interaction=with_inter)

        scaler = StandardScaler()
        logit = LogisticRegression(max_iter=2000, random_state=SEED)
        logit.fit(scaler.fit_transform(X_tr), y_tr)
        p_va = logit.predict_proba(scaler.transform(X_va))[:, 1]
        p_te = logit.predict_proba(scaler.transform(X_te))[:, 1]
        t = _best_threshold(y_va, p_va)
        esc = p_te >= t
        m = _metrics(y_te, p_te, esc.astype(int))
        g = score_policy(test, esc, cost_p95=cost_p95_test)
        m.update(
            {
                "config": "learned_logistic" + ("_interaction" if with_inter else ""),
                "with_interaction": with_inter,
                "threshold": t,
                "coverage": g["coverage"],
                "autonomy": g["autonomy"],
                "escalation_rate": g["escalation_rate"],
            }
        )
        rows.append(m)

        boost = lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.05, num_leaves=31,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        boost.fit(X_tr, y_tr)
        p_va = boost.predict_proba(X_va)[:, 1]
        p_te = boost.predict_proba(X_te)[:, 1]
        t = _best_threshold(y_va, p_va)
        esc = p_te >= t
        m = _metrics(y_te, p_te, esc.astype(int))
        g = score_policy(test, esc, cost_p95=cost_p95_test)
        name = "learned_lgbm" + ("_interaction" if with_inter else "")
        m.update(
            {
                "config": name,
                "with_interaction": with_inter,
                "threshold": t,
                "coverage": g["coverage"],
                "autonomy": g["autonomy"],
                "escalation_rate": g["escalation_rate"],
            }
        )
        rows.append(m)
        if not with_inter:
            esc_lgbm = esc

    th = quantiles_to_thresholds(val, Q_COST, Q_RISK)
    esc_dyn = escalate_dynamic_router(test, th)
    m_dyn = score_policy(test, esc_dyn, cost_p95=cost_p95_test)
    rows.append(
        {
            "config": "dynamic_router",
            "with_interaction": False,
            "threshold": np.nan,
            "accuracy": float(accuracy_score(y_te, esc_dyn.astype(int))),
            "f1": float(f1_score(y_te, esc_dyn.astype(int), zero_division=0)),
            "auc": float(roc_auc_score(y_te, esc_dyn.astype(float))),
            "ap": float(average_precision_score(y_te, esc_dyn.astype(float))),
            "coverage": m_dyn["coverage"],
            "autonomy": m_dyn["autonomy"],
            "escalation_rate": m_dyn["escalation_rate"],
        }
    )

    t_inter = float(
        np.quantile(
            val["cost"].to_numpy(dtype=float) * val["risk_score"].to_numpy(dtype=float),
            0.90,
        )
    )
    esc_inter = escalate_dynamic_with_interaction(test, th.t_cost, th.t_risk, t_inter)
    m_inter = score_policy(test, esc_inter, cost_p95=cost_p95_test)
    rows.append(
        {
            "config": "dynamic_router_interaction",
            "with_interaction": True,
            "threshold": t_inter,
            "accuracy": float(accuracy_score(y_te, esc_inter.astype(int))),
            "f1": float(f1_score(y_te, esc_inter.astype(int), zero_division=0)),
            "auc": float(roc_auc_score(y_te, esc_inter.astype(float))),
            "ap": float(average_precision_score(y_te, esc_inter.astype(float))),
            "coverage": m_inter["coverage"],
            "autonomy": m_inter["autonomy"],
            "escalation_rate": m_inter["escalation_rate"],
        }
    )

    cost = test["cost"].to_numpy(dtype=float)
    risk = test["risk_score"].to_numpy(dtype=float)
    med_c, med_r = float(np.median(cost)), float(np.median(risk))
    disagree = esc_lgbm != esc_dyn
    hi_c, hi_r = cost >= med_c, risk >= med_r
    region_counts = {
        "disagree_n": int(disagree.sum()),
        "disagree_hiCost_hiRisk": int((disagree & hi_c & hi_r).sum()),
        "disagree_hiCost_loRisk": int((disagree & hi_c & ~hi_r).sum()),
        "disagree_loCost_hiRisk": int((disagree & ~hi_c & hi_r).sum()),
        "disagree_loCost_loRisk": int((disagree & ~hi_c & ~hi_r).sum()),
    }

    out = pd.DataFrame(rows)
    for k, v in region_counts.items():
        out[k] = v
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "exp3.csv", index=False)

    print("=" * 72)
    print("EXP3 — Learned baseline vs threshold router (dual-axis oracle)")
    print("=" * 72)
    show = out[
        ["config", "with_interaction", "accuracy", "f1", "auc", "ap",
         "coverage", "autonomy", "escalation_rate"]
    ]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Disagreement (learned_lgbm vs dynamic_router):")
    for k, v in region_counts.items():
        print(f"  {k}: {v}")
    base = out.loc[out["config"] == "dynamic_router"].iloc[0]
    inter = out.loc[out["config"] == "dynamic_router_interaction"].iloc[0]
    lgbm_row = out.loc[out["config"] == "learned_lgbm"].iloc[0]
    print(
        f"Interaction closes gap? router AUC {base['auc']:.4f} -> {inter['auc']:.4f}; "
        f"learned_lgbm {lgbm_row['auc']:.4f} "
        f"(gap {lgbm_row['auc']-base['auc']:+.4f} -> {lgbm_row['auc']-inter['auc']:+.4f})"
    )
    print(f"wrote {RESULTS / 'exp3.csv'}")
    print("=" * 72)
    return out


if __name__ == "__main__":
    run_exp3()
