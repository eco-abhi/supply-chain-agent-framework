"""Additional experiments for paper/patent defensibility.

1. threshold_sensitivity  - sweep router thresholds, show the coverage/
   autonomy/latency tradeoff isn't an artifact of one chosen setting.
2. learned_baseline       - a from-scratch logistic regression trained
   against an oracle utility function, compared to the fixed-threshold
   router on the same oracle.
3. adversarial_injection  - a compromised agent that under-reports risk;
   shows statistical drift (PSI) alone misses it, while treating
   negotiation disagreement against an independent outcome signal
   catches it.
4. retrain_ablation       - calendar-based vs drift-triggered retraining
   after an injected shift; measures decisions-to-recovery.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

from .benchmark import LATENCY, decision_chain, is_high_stakes, simulate_decisions
from .drift import DriftMonitor
from .models import DecisionContext, GovernanceMode
from .router import GovernanceRouter, RouterConfig


# ------------------------------------------------------------------
# 1. Threshold sensitivity
# ------------------------------------------------------------------

def threshold_sensitivity(n: int = 1000, seed: int = 7):
    decisions = simulate_decisions(n, seed)
    cost_grid = [10_000, 25_000, 50_000, 75_000]
    risk_grid = [0.4, 0.5, 0.6, 0.7]
    rows = []
    for cost_t in cost_grid:
        for risk_t in risk_grid:
            router = GovernanceRouter(RouterConfig(cost_escalation_usd=cost_t,
                                                    risk_escalation=risk_t))
            hs_total = hs_gov = routine_total = routine_auto = 0
            lat = []
            for ctx in decisions:
                gov = router.route(ctx).governance
                l = LATENCY["a2a_negotiation"]
                if gov in (GovernanceMode.CENTRALIZED, GovernanceMode.HUMAN_APPROVAL):
                    l += LATENCY["orchestrator_hop"]
                if gov == GovernanceMode.HUMAN_APPROVAL:
                    l += LATENCY["human_gate"]
                lat.append(l)
                hs = ctx.cost_usd >= 25_000 or ctx.risk_score >= 0.6  # fixed def. of "high stakes"
                if hs:
                    hs_total += 1
                    hs_gov += gov != GovernanceMode.DECENTRALIZED
                else:
                    routine_total += 1
                    routine_auto += gov == GovernanceMode.DECENTRALIZED
            rows.append({
                "cost_threshold": cost_t, "risk_threshold": risk_t,
                "coverage": hs_gov / hs_total if hs_total else 1.0,
                "autonomy": routine_auto / routine_total if routine_total else 1.0,
                "latency": statistics.mean(lat),
            })
    return rows


# ------------------------------------------------------------------
# 2. Learned baseline vs fixed thresholds, against an oracle
# ------------------------------------------------------------------

def oracle_should_escalate(ctx: DecisionContext) -> int:
    """Ground truth used only to train/evaluate against: a nonlinear
    rule with an interaction term the linear threshold router can't
    represent exactly, so we can see how much that costs it."""
    score = (0.00002 * ctx.cost_usd) + (1.3 * ctx.risk_score) \
        + (0.5 * ctx.risk_score * (ctx.cost_usd / 100_000)) \
        + (0.15 * ctx.complexity)
    return 1 if score >= 1.0 else 0


def _sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def train_logistic(features, labels, lr: float = 0.1, epochs: int = 400):
    n_feat = len(features[0])
    w = [0.0] * n_feat
    b = 0.0
    n = len(features)
    for _ in range(epochs):
        grad_w = [0.0] * n_feat
        grad_b = 0.0
        for x, y in zip(features, labels):
            pred = _sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b)
            err = pred - y
            for i in range(n_feat):
                grad_w[i] += err * x[i]
            grad_b += err
        w = [wi - lr * (gw / n) for wi, gw in zip(w, grad_w)]
        b -= lr * (grad_b / n)
    return w, b


def learned_baseline(n: int = 1000, seed: int = 11, train_frac: float = 0.7):
    decisions = simulate_decisions(n, seed)
    # normalize features for stable training
    def feat(ctx):
        return [ctx.cost_usd / 100_000, ctx.risk_score, ctx.complexity / 3]

    labels = [oracle_should_escalate(c) for c in decisions]
    split = int(len(decisions) * train_frac)
    train_x = [feat(c) for c in decisions[:split]]
    train_y = labels[:split]
    test_ctx = decisions[split:]
    test_y = labels[split:]

    w, b = train_logistic(train_x, train_y)

    def learned_predict(ctx):
        return 1 if _sigmoid(sum(wi * xi for wi, xi in zip(w, feat(ctx))) + b) >= 0.5 else 0

    router = GovernanceRouter(RouterConfig())

    def router_predict(ctx):
        return 1 if router.route(ctx).governance != GovernanceMode.DECENTRALIZED else 0

    def accuracy(pred_fn):
        correct = sum(pred_fn(c) == y for c, y in zip(test_ctx, test_y))
        return correct / len(test_ctx)

    def f1(pred_fn):
        tp = fp = fn = 0
        for c, y in zip(test_ctx, test_y):
            p = pred_fn(c)
            tp += p == 1 and y == 1
            fp += p == 1 and y == 0
            fn += p == 0 and y == 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "router_accuracy": accuracy(router_predict),
        "router_f1": f1(router_predict),
        "learned_accuracy": accuracy(learned_predict),
        "learned_f1": f1(learned_predict),
        "learned_weights": {"cost": w[0], "risk": w[1], "complexity": w[2], "bias": b},
    }


# ------------------------------------------------------------------
# 3. Adversarial injection: compromised agent under-reporting risk
# ------------------------------------------------------------------

def adversarial_injection(n: int = 600, seed: int = 5, incident_scale: float = 2.0):
    """A compromised (or simply broken) supplier-risk agent whose reports
    are drawn from the *correct marginal distribution* of risk scores, but
    are decorrelated from the actual decision, an assignment failure
    rather than a distributional shift. Statistical drift monitoring
    (PSI) compares only the marginal distribution of reported values, so
    it is structurally blind to this: the histogram looks identical to
    the healthy reference. An outcome-based disagreement check, which
    compares each report against what actually happened for that specific
    decision, catches it directly."""
    rng = random.Random(seed)
    reference = [min(1.0, max(0.0, rng.gauss(0.3, 0.15))) for _ in range(300)]
    psi_monitor = DriftMonitor("psi_only", reference=list(reference))
    disagreement_monitor = DriftMonitor("disagreement", reference=list(reference))

    detections_psi_at = None
    detections_disagreement_at = None
    hidden_incidents = 0

    for i in range(n):
        true_risk = min(1.0, max(0.0, rng.gauss(0.3, 0.15)))
        # compromised agent: reports a value from the SAME right marginal
        # distribution, but uncorrelated with this decision's true risk
        reported_risk = min(1.0, max(0.0, rng.gauss(0.3, 0.15)))
        psi_monitor.observe(reported_risk)

        # independent outcome signal: incident probability tracks TRUE
        # risk, not the (decorrelated) reported risk
        incident = rng.random() < min(1.0, true_risk * incident_scale)
        reported_low = reported_risk < 0.4
        agreed = not (reported_low and incident)
        if reported_low and incident:
            hidden_incidents += 1
        disagreement_monitor.record_negotiation(agreed=agreed)

        if detections_psi_at is None and psi_monitor.drifting:
            detections_psi_at = i
        if detections_disagreement_at is None and disagreement_monitor.drifting:
            detections_disagreement_at = i

    return {
        "n": n,
        "final_psi": psi_monitor.current_psi,
        "psi_detected": detections_psi_at is not None,
        "psi_detected_at": detections_psi_at,
        "disagreement_detected": detections_disagreement_at is not None,
        "disagreement_detected_at": detections_disagreement_at,
        "hidden_incidents": hidden_incidents,
    }


# ------------------------------------------------------------------
# 4. Retraining ablation: calendar vs drift-triggered
# ------------------------------------------------------------------

def retrain_ablation(n: int = 1000, seed: int = 3, shift_at: int = 300,
                     shift_size: float = 0.35, calendar_interval: int = 250):
    """Compares calendar-based vs drift-triggered retraining after an
    injected distributional shift.

    The key design point: the router acts on the model's PREDICTION of
    risk (its last-trained mean), not on ground truth directly. Before
    retraining, a stale model keeps predicting the old, lower risk level
    even though the true underlying risk has shifted, so decisions get
    under-governed until the model catches up. Both modes rebaseline the
    monitor identically on retrain (same mechanic); they differ only in
    *when* that trigger fires. shift_at is deliberately not aligned to a
    calendar checkpoint, so the calendar schedule's lag depends on how
    unlucky the timing is, exactly the failure mode it's prone to in
    practice, since the real world doesn't wait for your retrain cron job."""
    rng = random.Random(seed)
    reference = [rng.gauss(0.3, 0.1) for _ in range(300)]

    def run(mode: str):
        monitor = DriftMonitor("model", reference=list(reference))
        router = GovernanceRouter(RouterConfig())
        model_mean = 0.3          # the model's current (possibly stale) belief
        mis_governed = 0          # true-high-risk decisions the router didn't escalate
        recovery_at = None
        for i in range(n):
            shifted = i >= shift_at
            true_mean = 0.3 + (shift_size if shifted else 0.0)
            true_risk = min(1.0, max(0.0, rng.gauss(true_mean, 0.1)))
            monitor.observe(true_risk)   # drift detector sees ground truth

            retrain_now = (
                (mode == "calendar" and i > 0 and i % calendar_interval == 0)
                or (mode == "drift" and monitor.drifting)
            )
            if retrain_now:
                if monitor.last_batch_mean is not None:
                    model_mean = monitor.last_batch_mean
                if shifted and recovery_at is None:
                    recovery_at = i - shift_at
                monitor.retrain()

            # the router sees the model's (possibly stale) prediction, not
            # truth. model_drift_flag is held constant (False) here so this
            # ablation isolates the retrain-timing effect on model accuracy
            # alone, rather than mixing in the router's separate drift-bump
            # bonus, which both modes could equally receive in practice.
            predicted_risk = min(1.0, max(0.0, rng.gauss(model_mean, 0.1)))
            ctx = DecisionContext(cost_usd=5_000, risk_score=predicted_risk,
                                  complexity=2, model_drift_flag=False)
            gov = router.route(ctx).governance
            if shifted and true_risk >= 0.6 and gov == GovernanceMode.DECENTRALIZED:
                mis_governed += 1

        return {"mode": mode, "mis_governed_high_risk_decisions": mis_governed,
                "recovery_lag_decisions": recovery_at}

    return [run("calendar"), run("drift")]


if __name__ == "__main__":
    print("=== 1. Threshold sensitivity ===")
    for row in threshold_sensitivity():
        print(f"  cost>=${row['cost_threshold']:>6,} risk>={row['risk_threshold']:.1f} "
              f"-> coverage {row['coverage']:.0%}, autonomy {row['autonomy']:.0%}, "
              f"latency {row['latency']:.2f}")

    print("\n=== 2. Learned baseline vs fixed-threshold router ===")
    lb = learned_baseline()
    print(f"  router:  accuracy {lb['router_accuracy']:.1%}  f1 {lb['router_f1']:.3f}")
    print(f"  learned: accuracy {lb['learned_accuracy']:.1%}  f1 {lb['learned_f1']:.3f}")
    print(f"  learned weights: {lb['learned_weights']}")

    print("\n=== 3. Adversarial injection (compromised risk agent) ===")
    ai = adversarial_injection()
    print(f"  PSI-only detection:          {'YES at ' + str(ai['psi_detected_at']) if ai['psi_detected'] else 'NOT DETECTED'} "
          f"(final PSI {ai['final_psi']:.3f})")
    print(f"  Disagreement-based detection: {'YES at ' + str(ai['disagreement_detected_at']) if ai['disagreement_detected'] else 'NOT DETECTED'}")
    print(f"  Decisions where a low report hid a real incident: {ai['hidden_incidents']}")
    psi_hits = sum(adversarial_injection(seed=s)["psi_detected"] for s in range(1, 31))
    dis_hits = sum(adversarial_injection(seed=s)["disagreement_detected"] for s in range(1, 31))
    print(f"  Across 30 seeds: PSI detected in {psi_hits}/30 runs, disagreement detected in {dis_hits}/30 runs")

    print("\n=== 4. Retraining ablation (calendar vs drift-triggered) ===")
    for r in retrain_ablation():
        lag = r["recovery_lag_decisions"]
        print(f"  {r['mode']:<10} mis-governed high-risk decisions after shift: "
              f"{r['mis_governed_high_risk_decisions']:>4}  "
              f"recovery lag: {lag if lag is not None else 'n/a (fixed interval)'}")
