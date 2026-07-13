"""Regression tests for the paper/patent defensibility experiments.
These pin down the qualitative findings, not exact numbers, since the
underlying simulations are stochastic.
"""
from scaf.experiments import (
    adversarial_injection, learned_baseline, retrain_ablation,
    threshold_sensitivity,
)


def test_threshold_sensitivity_covers_grid():
    rows = threshold_sensitivity(n=200)
    assert len(rows) == 16  # 4 cost thresholds x 4 risk thresholds
    for row in rows:
        assert 0.0 <= row["coverage"] <= 1.0
        assert 0.0 <= row["autonomy"] <= 1.0
    # tighter thresholds should never give strictly less coverage than
    # looser ones, holding risk threshold fixed
    by_risk = {}
    for row in rows:
        by_risk.setdefault(row["risk_threshold"], []).append(row)
    for risk_t, group in by_risk.items():
        group.sort(key=lambda r: r["cost_threshold"])
        coverages = [g["coverage"] for g in group]
        assert coverages[0] >= coverages[-1] - 1e-9


def test_learned_baseline_runs_and_scores_reasonably():
    result = learned_baseline(n=500)
    assert 0.0 <= result["router_accuracy"] <= 1.0
    assert 0.0 <= result["learned_accuracy"] <= 1.0
    # both should clearly beat a coin flip against the oracle
    assert result["router_accuracy"] > 0.6
    assert result["learned_accuracy"] > 0.6


def test_adversarial_injection_psi_misses_disagreement_catches():
    """The core defensibility claim: a decorrelated-report failure is
    structurally invisible to marginal-distribution drift checks, but
    caught by outcome-based disagreement. Verified across seeds since
    any single run is stochastic."""
    psi_hits = sum(adversarial_injection(seed=s)["psi_detected"] for s in range(1, 21))
    dis_hits = sum(adversarial_injection(seed=s)["disagreement_detected"] for s in range(1, 21))
    assert psi_hits <= 3          # near the false-positive floor, not real detection
    assert dis_hits >= 18         # reliably caught


def test_retrain_ablation_drift_recovers_faster():
    """Drift-triggered retraining should recover faster and mis-govern
    fewer high-risk decisions than a calendar schedule misaligned with
    the actual shift, averaged over seeds to avoid one lucky draw."""
    cal_mis, drift_mis = [], []
    for seed in range(1, 11):
        cal, drift = retrain_ablation(seed=seed)
        cal_mis.append(cal["mis_governed_high_risk_decisions"])
        drift_mis.append(drift["mis_governed_high_risk_decisions"])
        assert drift["recovery_lag_decisions"] <= cal["recovery_lag_decisions"]
    assert sum(drift_mis) < sum(cal_mis)
