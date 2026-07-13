"""Guardrails: policy-as-code, enforced deterministically.

These are hard limits, not prompts. An agent cannot argue its way past them.
Covers Section 3.4 of the paper: action boundaries, tool scoping,
inter-agent message validation, and mandatory audit logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import AuditRecord


class GuardrailViolation(Exception):
    pass


@dataclass
class Policy:
    """Hard action boundaries per agent role."""
    max_po_approval_usd: dict = field(default_factory=lambda: {
        "finance": 250_000.0,
        "demand": 0.0,        # demand agent can never approve POs directly
        "supplier_risk": 0.0,
        "logistics": 0.0,
    })
    # per-agent MCP tool scoping: which tools each agent may call
    tool_scope: dict = field(default_factory=lambda: {
        "demand": {"get_inventory", "get_sales_history", "get_forecast"},
        "supplier_risk": {"get_supplier_financials", "get_delivery_history"},
        "finance": {"get_budget", "get_po_history", "approve_po"},
        "logistics": {"get_carriers", "get_shipment_status"},
        "orchestrator": {"*"},
    })


class GuardrailEngine:
    def __init__(self, policy: Policy | None = None):
        self.policy = policy or Policy()
        self.audit_log: list[AuditRecord] = []

    # -- 1. hard action boundaries --
    def check_po_approval(self, agent: str, amount_usd: float) -> None:
        ceiling = self.policy.max_po_approval_usd.get(agent, 0.0)
        if amount_usd > ceiling:
            raise GuardrailViolation(
                f"{agent} cannot approve PO of ${amount_usd:,.0f} "
                f"(ceiling ${ceiling:,.0f}); requires escalation"
            )

    # -- 2. tool scoping --
    def check_tool_access(self, agent: str, tool: str) -> None:
        scope = self.policy.tool_scope.get(agent, set())
        if "*" not in scope and tool not in scope:
            raise GuardrailViolation(f"{agent} is not scoped for tool '{tool}'")

    # -- 3. inter-agent message validation --
    def validate_a2a_message(self, sender: str, payload: dict) -> None:
        # basic structural + range validation; extend per domain
        if "risk" in payload:
            r = payload["risk"]
            if not (isinstance(r, (int, float)) and 0.0 <= r <= 1.0):
                raise GuardrailViolation(
                    f"invalid risk value {r!r} in A2A message from {sender}"
                )
        if "amount_usd" in payload and payload["amount_usd"] < 0:
            raise GuardrailViolation(
                f"negative amount in A2A message from {sender}"
            )

    # -- 4. mandatory audit logging --
    def log(self, record: AuditRecord) -> None:
        self.audit_log.append(record)

    def explain(self, decision_id: str) -> list[AuditRecord]:
        """Instant reporting: reconstruct the full decision chain on demand."""
        return [r for r in self.audit_log if r.decision_id == decision_id]
