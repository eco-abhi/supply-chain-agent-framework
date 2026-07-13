"""Benchmark harness (Section 5 of the paper, runnable).

Compares three configurations over N simulated decisions:
  A. static centralized  (everything escalates to the orchestrator)
  B. static decentralized (everything resolves peer-to-peer)
  C. dynamic router       (the framework)

Metrics: mean decision latency (simulated), token cost (SCQL vs JSON),
governance coverage (share of high-stakes decisions that got oversight),
and autonomy (share of routine decisions that avoided escalation).
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

from .models import DecisionContext, GovernanceMode
from .router import GovernanceRouter, RouterConfig
from .scql import Chain

# simulated latency costs (arbitrary units; relative values are what matter)
LATENCY = {
    "a2a_negotiation": 1.0,
    "orchestrator_hop": 2.5,   # central queueing + global state check
    "human_gate": 20.0,        # human in the loop is slow
}

HIGH_STAKES_COST = 25_000.0
HIGH_STAKES_RISK = 0.6


@dataclass
class RunResult:
    name: str
    mean_latency: float
    governance_coverage: float   # high-stakes decisions that received oversight
    autonomy: float              # routine decisions resolved without escalation
    scql_tokens: int
    json_tokens: int


def simulate_decisions(n: int, seed: int) -> list[DecisionContext]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        # log-ish spread of order sizes, occasional big ones
        cost = rng.choice([500, 2_000, 8_000, 15_000, 40_000, 120_000])
        risk = min(1.0, max(0.0, rng.gauss(0.35, 0.2)))
        out.append(DecisionContext(
            description=f"decision_{i}", cost_usd=cost, risk_score=risk,
            complexity=rng.choice([1, 2, 2, 3]),
        ))
    return out


def is_high_stakes(ctx: DecisionContext) -> bool:
    return ctx.cost_usd >= HIGH_STAKES_COST or ctx.risk_score >= HIGH_STAKES_RISK


def decision_chain(ctx: DecisionContext) -> Chain:
    """Build the message chain for one decision, for token accounting."""
    ch = Chain(ctx.decision_id)
    ch.define("P1", "Part", id=4471, name="BatteryModuleHousing")
    ch.define("S1", "Supplier", id="ABC001", name="ABCCorp")
    ch.emit("MCP", "get", "P1.stock")
    ch.emit("A2A", "ask", "supplier_risk", "S1 risk?")
    ch.emit("A2A", "tell", "demand", f"S1 risk:{ctx.risk_score:.2f}")
    return ch


def run_config(name: str, decisions: list[DecisionContext],
               mode: str) -> RunResult:
    router = GovernanceRouter(RouterConfig())
    latencies, scql_total, json_total = [], 0, 0
    hs_total = hs_governed = routine_total = routine_autonomous = 0

    for ctx in decisions:
        ch = decision_chain(ctx)
        scql_total += ch.scql_tokens()
        json_total += ch.json_equivalent_tokens()

        if mode == "centralized":
            gov = GovernanceMode.HUMAN_APPROVAL if ctx.cost_usd >= 100_000 else GovernanceMode.CENTRALIZED
        elif mode == "decentralized":
            gov = GovernanceMode.DECENTRALIZED
        else:  # dynamic
            gov = router.route(ctx).governance

        lat = LATENCY["a2a_negotiation"]
        if gov in (GovernanceMode.CENTRALIZED, GovernanceMode.HUMAN_APPROVAL):
            lat += LATENCY["orchestrator_hop"]
        if gov == GovernanceMode.HUMAN_APPROVAL:
            lat += LATENCY["human_gate"]
        latencies.append(lat)

        if is_high_stakes(ctx):
            hs_total += 1
            if gov != GovernanceMode.DECENTRALIZED:
                hs_governed += 1
        else:
            routine_total += 1
            if gov == GovernanceMode.DECENTRALIZED:
                routine_autonomous += 1

    return RunResult(
        name=name,
        mean_latency=statistics.mean(latencies),
        governance_coverage=hs_governed / hs_total if hs_total else 1.0,
        autonomy=routine_autonomous / routine_total if routine_total else 1.0,
        scql_tokens=scql_total,
        json_tokens=json_total,
    )


def run_benchmark(n: int = 1000, seed: int = 7) -> list[RunResult]:
    decisions = simulate_decisions(n, seed)
    return [
        run_config("static centralized", decisions, "centralized"),
        run_config("static decentralized", decisions, "decentralized"),
        run_config("dynamic router", decisions, "dynamic"),
    ]


if __name__ == "__main__":
    results = run_benchmark()
    print(f"{'config':<24}{'latency':>9}{'gov cover':>11}{'autonomy':>10}{'SCQL tok':>10}{'JSON tok':>10}")
    for r in results:
        print(f"{r.name:<24}{r.mean_latency:>9.2f}{r.governance_coverage:>10.0%}"
              f"{r.autonomy:>10.0%}{r.scql_tokens:>10,}{r.json_tokens:>10,}")
