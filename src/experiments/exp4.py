"""exp4 — Adversarial robustness: rolling-PSI vs disagreement monitors."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..eval_data import build_locked_decisions, split_frames
from ..monitors import population_stability_index, two_proportion_ztest, wilson_ci

SEED = 42
N_SEEDS = 30
PSI_THRESHOLD = 0.2
REF_WINDOW = 1000          # rolling reference of recent reported scores
LIVE_BATCH = 300           # tuned so honest FPR ≈ 0.03–0.05 at threshold 0.2
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
LOW_RISK_Q = 0.30


def _disagreement_flag(
    risk_reported: np.ndarray,
    adverse: np.ndarray,
    *,
    t_low: float,
    alpha: float = 0.05,
) -> tuple[bool, float, float, float]:
    """Among reported low-risk: realized adverse vs mean reported score."""
    low = risk_reported <= t_low
    n = int(low.sum())
    if n < 30:
        return False, float("nan"), float("nan"), float("nan")
    k = int(adverse[low].sum())
    p_hat = k / n
    expected = float(np.clip(risk_reported[low].mean(), 1e-6, 1 - 1e-6))
    se = np.sqrt(expected * (1 - expected) / n)
    z = (p_hat - expected) / se if se > 0 else 0.0
    pval = float(norm.sf(z))
    return bool(pval < alpha and p_hat > expected), float(p_hat), float(expected), float(pval)


def rolling_psi_stream(
    reported: np.ndarray,
    *,
    ref_window: int = REF_WINDOW,
    live_batch: int = LIVE_BATCH,
    threshold: float = PSI_THRESHOLD,
) -> dict:
    """Stream reported scores; PSI(live batch vs prior rolling window).

    Returns whether any post-warmup batch exceeded ``threshold``, plus the
    list of per-batch PSI values (for diagnosis when FPR is off-target).
    """
    reported = np.asarray(reported, dtype=float)
    n = len(reported)
    if n < ref_window + live_batch:
        return {"detect": False, "psi_values": [], "n_checks": 0, "max_psi": 0.0}

    psi_values: list[float] = []
    detect = False
    # cursor starts after warm-up reference
    cursor = ref_window
    while cursor + live_batch <= n:
        ref = reported[cursor - ref_window : cursor]
        live = reported[cursor : cursor + live_batch]
        psi = population_stability_index(ref, live)
        psi_values.append(float(psi))
        if psi > threshold:
            detect = True
        cursor += live_batch

    return {
        "detect": detect,
        "psi_values": psi_values,
        "n_checks": len(psi_values),
        "max_psi": float(max(psi_values)) if psi_values else 0.0,
        "mean_psi": float(np.mean(psi_values)) if psi_values else 0.0,
    }


def run_exp4() -> pd.DataFrame:
    df = build_locked_decisions()
    frames = split_frames(df)
    val, test = frames["validation"], frames["test"]
    t_low = float(val["risk_score"].quantile(LOW_RISK_Q))

    true_risk = test["risk_score"].to_numpy(dtype=float)
    adverse = test["adverse_outcome"].to_numpy(dtype=int)
    n = len(true_risk)

    # ------------------------------------------------------------------
    # 1) Honest baseline FPR (30 seeds) — BEFORE attack
    #    Each seed shuffles the temporal order of honest (score, label)
    #    pairs so the rolling PSI sees a stationary honest stream with
    #    seed-specific ordering (marginal preserved).
    # ------------------------------------------------------------------
    honest_rows = []
    psi_fpr = 0
    dis_fpr = 0
    for i in range(N_SEEDS):
        rng = np.random.default_rng(SEED + 100 + i)
        order = rng.permutation(n)
        scores_h = true_risk[order]
        labels_h = adverse[order]
        psi_res = rolling_psi_stream(scores_h)
        dis_flag, rate, expected, pval = _disagreement_flag(
            scores_h, labels_h, t_low=t_low
        )
        psi_fpr += int(psi_res["detect"])
        dis_fpr += int(dis_flag)
        honest_rows.append(
            {
                "seed": i,
                "condition": "honest",
                "psi_detect": int(psi_res["detect"]),
                "psi_max": psi_res["max_psi"],
                "psi_mean": psi_res["mean_psi"],
                "psi_n_checks": psi_res["n_checks"],
                "disagreement_detect": int(dis_flag),
                "low_risk_adverse_rate": rate,
                "expected_from_reported": expected,
                "disagreement_pval": pval,
                "t_low": t_low,
            }
        )

    psi_fpr_rate = psi_fpr / N_SEEDS
    print("=" * 72)
    print("EXP4 — honest baseline FPR (print BEFORE attack)")
    print("=" * 72)
    print(
        f"rolling PSI: ref_window={REF_WINDOW}, live_batch={LIVE_BATCH}, "
        f"threshold={PSI_THRESHOLD}"
    )
    print(
        f"PSI honest FPR:          {psi_fpr}/{N_SEEDS}  "
        f"rate={psi_fpr_rate:.3f}  "
        f"Wilson95={wilson_ci(psi_fpr, N_SEEDS)}"
    )
    print(
        f"Disagreement honest FPR: {dis_fpr}/{N_SEEDS}  "
        f"rate={dis_fpr/N_SEEDS:.3f}  "
        f"Wilson95={wilson_ci(dis_fpr, N_SEEDS)}"
    )
    if not (0.02 <= psi_fpr_rate <= 0.08):
        honest_psis = [r["psi_max"] for r in honest_rows]
        print(
            "WARNING: PSI FPR not in ~0.03–0.05 band. "
            f"honest max-PSI per seed: "
            f"mean={np.mean(honest_psis):.4f}  "
            f"median={np.median(honest_psis):.4f}  "
            f"p90={np.percentile(honest_psis, 90):.4f}  "
            f"max={np.max(honest_psis):.4f}"
        )
        print(
            "  per-seed max PSI: "
            + ", ".join(f"{v:.3f}" for v in honest_psis)
        )
    else:
        print("PSI honest FPR is in the target ~0.03–0.05 band — proceeding to attack.")

    # ------------------------------------------------------------------
    # 2) Permutation attack (30 seeds)
    # ------------------------------------------------------------------
    attack_rows = []
    psi_det = 0
    dis_det = 0
    for i in range(N_SEEDS):
        rng = np.random.default_rng(SEED + 1000 + i)
        compromised = rng.permutation(true_risk)
        psi_res = rolling_psi_stream(compromised)
        dis_flag, rate, expected, pval = _disagreement_flag(
            compromised, adverse, t_low=t_low
        )
        psi_det += int(psi_res["detect"])
        dis_det += int(dis_flag)
        attack_rows.append(
            {
                "seed": i,
                "condition": "attack",
                "psi_detect": int(psi_res["detect"]),
                "psi_max": psi_res["max_psi"],
                "psi_mean": psi_res["mean_psi"],
                "psi_n_checks": psi_res["n_checks"],
                "disagreement_detect": int(dis_flag),
                "low_risk_adverse_rate": rate,
                "expected_from_reported": expected,
                "disagreement_pval": pval,
                "t_low": t_low,
            }
        )

    out = pd.DataFrame(honest_rows + attack_rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "exp4.csv", index=False)

    psi_ci = wilson_ci(psi_det, N_SEEDS)
    dis_ci = wilson_ci(dis_det, N_SEEDS)
    psi_fpr_ci = wilson_ci(psi_fpr, N_SEEDS)
    dis_fpr_ci = wilson_ci(dis_fpr, N_SEEDS)
    z_psi, p_psi = two_proportion_ztest(psi_det, N_SEEDS, psi_fpr, N_SEEDS)
    z_dis, p_dis = two_proportion_ztest(dis_det, N_SEEDS, dis_fpr, N_SEEDS)

    print()
    print("--- attack results ---")
    print(
        f"PSI attack detections:          {psi_det}/{N_SEEDS}  "
        f"rate={psi_det/N_SEEDS:.3f}  Wilson95=[{psi_ci[0]:.3f}, {psi_ci[1]:.3f}]"
    )
    print(
        f"PSI honest FPR:                 {psi_fpr}/{N_SEEDS}  "
        f"rate={psi_fpr/N_SEEDS:.3f}  Wilson95=[{psi_fpr_ci[0]:.3f}, {psi_fpr_ci[1]:.3f}]"
    )
    print(
        f"Two-proportion (PSI attack vs FPR): z={z_psi:.3f}  p={p_psi:.4g}"
    )
    print(
        f"Disagreement attack detections: {dis_det}/{N_SEEDS}  "
        f"rate={dis_det/N_SEEDS:.3f}  Wilson95=[{dis_ci[0]:.3f}, {dis_ci[1]:.3f}]"
    )
    print(
        f"Disagreement honest FPR:        {dis_fpr}/{N_SEEDS}  "
        f"rate={dis_fpr/N_SEEDS:.3f}  Wilson95=[{dis_fpr_ci[0]:.3f}, {dis_fpr_ci[1]:.3f}]"
    )
    print(
        f"Two-proportion (disagreement attack vs FPR): z={z_dis:.3f}  p={p_dis:.4g}"
    )
    print(f"wrote {RESULTS / 'exp4.csv'}")
    print("=" * 72)
    return out


if __name__ == "__main__":
    run_exp4()
