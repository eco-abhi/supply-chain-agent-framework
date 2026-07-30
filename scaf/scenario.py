"""The worked example from the paper (Section 4), runnable.

A part shortage is detected; the router decides per decision whether the
response stays decentralized or escalates, and everything is auditable
after the fact.
"""
from __future__ import annotations

import random

from .agents.core import DemandAgent, FinanceAgent, Orchestrator, SupplierRiskAgent
from .guardrails import GuardrailEngine
from .models import DecisionContext, GovernanceMode
from .router import GovernanceRouter, RouterConfig


def run_stockout_scenario(seed: int = 42, order_cost: float = 12_000.0,
                          inject_drift: float = 0.0, verbose: bool = True):
    rng = random.Random(seed)
    guardrails = GuardrailEngine()
    router = GovernanceRouter(RouterConfig())

    demand = DemandAgent(guardrails, rng)
    risk_agent = SupplierRiskAgent(guardrails, rng)
    finance = FinanceAgent(guardrails)
    orchestrator = Orchestrator(guardrails, finance)

    if inject_drift:
        risk_agent.inject_drift(inject_drift)
        # warm the live window so PSI has data
        for _ in range(60):
            risk_agent.monitor.observe(min(1.0, max(0.0, rng.gauss(0.3 + inject_drift, 0.1))))

    ctx = DecisionContext(description="stockout response P1", cost_usd=order_cost)

    # 1. demand agent checks inventory (MCP)
    inv = demand.check_inventory("P1", ctx)
    if not inv["short"]:
        if verbose:
            print("No shortage detected; nothing to do.")
        return guardrails, None

    # 2. demand agent asks supplier risk agent (A2A)
    risk = risk_agent.assess("S1", ctx)
    guardrails.validate_a2a_message("supplier_risk", {"risk": risk})
    ctx.risk_score = risk
    ctx.model_drift_flag = risk_agent.monitor.drifting
    ctx.complexity = 2

    # 3. router decides
    routing = router.route(ctx)
    if verbose:
        print(f"Routing: {routing.protocol.value} / {routing.governance.value}")
        print(f"  because {routing.reason}")

    # 4. execute per governance mode
    if routing.governance == GovernanceMode.DECENTRALIZED:
        approved = finance.approve(ctx.cost_usd, ctx)
    else:
        approved = orchestrator.handle(ctx, routing.governance, ctx.cost_usd)

    if verbose:
        print(f"Order {'placed' if approved else 'blocked'}.")
        print(f"\nAudit trail for decision {ctx.decision_id}:")
        for rec in guardrails.explain(ctx.decision_id):
            print(f"  [{rec.actor}] {rec.action}: {rec.rationale}")

    return guardrails, routing
