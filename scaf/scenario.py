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
from .scql import Chain


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

    chain = Chain("c1")
    chain.define("P1", "Part", id=4471, name="BatteryModuleHousing")
    chain.define("S1", "Supplier", id="ABC001", name="ABCCorp")

    ctx = DecisionContext(description="stockout response P1", cost_usd=order_cost)

    # 1. demand agent checks inventory (MCP)
    chain.emit("MCP", "get", "P1.stock")
    inv = demand.check_inventory("P1", ctx)
    if not inv["short"]:
        if verbose:
            print("No shortage detected; nothing to do.")
        return guardrails, chain, None

    # 2. demand agent asks supplier risk agent (A2A)
    chain.emit("A2A", "ask", "supplier_risk", "S1 risk?")
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
        chain.emit("A2A", "tell", "demand", f"S1 risk:{risk:.2f} -> reorder P1")
        approved = finance.approve(ctx.cost_usd, ctx)
    else:
        chain.emit("A2A", "ask", "orchestrator", f"escalate P1 reorder ${order_cost:,.0f}")
        approved = orchestrator.handle(ctx, routing.governance, ctx.cost_usd)

    if verbose:
        print(f"Order {'placed' if approved else 'blocked'}.")
        print(f"\nSCQL tokens: {chain.scql_tokens()}  |  JSON-equivalent: {chain.json_equivalent_tokens()}")
        print(f"\nAudit trail for decision {ctx.decision_id}:")
        for rec in guardrails.explain(ctx.decision_id):
            print(f"  [{rec.actor}] {rec.action}: {rec.rationale}")

    return guardrails, chain, routing
