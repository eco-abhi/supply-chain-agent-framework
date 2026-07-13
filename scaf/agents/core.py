"""Domain agents and the central orchestrator.

Simplified reference implementations: each agent owns a model stub
(risk scoring, forecasting) and a DriftMonitor, and communicates through
the protocol layer. The orchestrator is the centralized path the router
escalates to.
"""
from __future__ import annotations

import random

from ..drift import DriftMonitor
from ..guardrails import GuardrailEngine, GuardrailViolation
from ..models import AuditRecord, DecisionContext, GovernanceMode, Protocol


class SupplierRiskAgent:
    def __init__(self, guardrails: GuardrailEngine, rng: random.Random):
        self.name = "supplier_risk"
        self.guardrails = guardrails
        self.rng = rng
        # reference distribution of historical risk scores
        self.monitor = DriftMonitor(
            "supplier_risk_model",
            reference=[rng.gauss(0.3, 0.1) for _ in range(200)],
        )
        self._shift = 0.0  # injected drift for benchmarks

    def inject_drift(self, shift: float) -> None:
        self._shift = shift

    def assess(self, supplier_id: str, ctx: DecisionContext) -> float:
        score = min(1.0, max(0.0, self.rng.gauss(0.3 + self._shift, 0.1)))
        self.monitor.observe(score)
        self.guardrails.log(AuditRecord(
            decision_id=ctx.decision_id, actor=self.name,
            action=f"assess({supplier_id})",
            rationale=f"risk={score:.2f}, psi={self.monitor.current_psi:.3f}",
            protocol=Protocol.A2A,
        ))
        return score


class DemandAgent:
    def __init__(self, guardrails: GuardrailEngine, rng: random.Random):
        self.name = "demand"
        self.guardrails = guardrails
        self.rng = rng

    def check_inventory(self, part_id: str, ctx: DecisionContext) -> dict:
        self.guardrails.check_tool_access(self.name, "get_inventory")
        stock = self.rng.randint(100, 600)
        threshold = 500
        self.guardrails.log(AuditRecord(
            decision_id=ctx.decision_id, actor=self.name,
            action=f"get_inventory({part_id})",
            rationale=f"stock={stock}, reorder_threshold={threshold}",
            protocol=Protocol.MCP,
        ))
        return {"stock": stock, "threshold": threshold, "short": stock < threshold}


class FinanceAgent:
    def __init__(self, guardrails: GuardrailEngine):
        self.name = "finance"
        self.guardrails = guardrails

    def approve(self, amount_usd: float, ctx: DecisionContext) -> bool:
        try:
            self.guardrails.check_po_approval(self.name, amount_usd)
        except GuardrailViolation as e:
            self.guardrails.log(AuditRecord(
                decision_id=ctx.decision_id, actor=self.name,
                action="approve_po", rationale=f"BLOCKED: {e}",
            ))
            return False
        self.guardrails.log(AuditRecord(
            decision_id=ctx.decision_id, actor=self.name,
            action="approve_po", rationale=f"approved ${amount_usd:,.0f} within ceiling",
        ))
        return True


class Orchestrator:
    """The centralized path. Holds global state, enforces rules, and
    (in HUMAN_APPROVAL mode) gates on sign-off."""

    def __init__(self, guardrails: GuardrailEngine, finance: FinanceAgent):
        self.name = "orchestrator"
        self.guardrails = guardrails
        self.finance = finance

    def handle(self, ctx: DecisionContext, governance: GovernanceMode,
               amount_usd: float, human_approves: bool = True) -> bool:
        if governance == GovernanceMode.HUMAN_APPROVAL:
            self.guardrails.log(AuditRecord(
                decision_id=ctx.decision_id, actor=self.name,
                action="human_gate",
                rationale=f"human sign-off {'granted' if human_approves else 'denied'}",
                governance=governance,
            ))
            if not human_approves:
                return False
        approved = self.finance.approve(amount_usd, ctx)
        self.guardrails.log(AuditRecord(
            decision_id=ctx.decision_id, actor=self.name,
            action="centralized_resolution",
            rationale=f"finance {'approved' if approved else 'blocked'}",
            governance=governance,
        ))
        return approved
