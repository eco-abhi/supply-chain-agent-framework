"""The governance router: the framework's core mechanism.

Jointly selects (protocol, governance mode) per decision, based on
risk score, cost threshold, complexity, and model drift signals.

This is the piece described in Section 3.2 of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import DecisionContext, GovernanceMode, Protocol, RoutingResult


@dataclass
class RouterConfig:
    # governance thresholds
    cost_escalation_usd: float = 25_000.0    # at/above -> centralized
    cost_human_approval_usd: float = 100_000.0  # at/above -> human sign-off
    risk_escalation: float = 0.6             # at/above -> centralized
    # protocol thresholds
    complexity_a2a_min: int = 2              # >= this many agents/sources -> A2A
    # drift handling
    drift_risk_bump: float = 0.2             # added to risk score when a model is drifting


class GovernanceRouter:
    """Routes each decision to a (protocol, governance mode) pair.

    Protocol selection follows the empirical MCP/A2A tradeoff:
      - single-source stateless lookup        -> HTTP
      - single-source contextual data access  -> MCP
      - multi-agent negotiation/delegation    -> A2A

    Governance selection is threshold-based on effective risk and cost:
      - below thresholds  -> decentralized (autonomous A2A between peers)
      - above either      -> centralized (orchestrator holds state, enforces rules)
      - above human gate  -> centralized + human approval required
    """

    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()

    def effective_risk(self, ctx: DecisionContext) -> float:
        risk = ctx.risk_score
        if ctx.model_drift_flag:
            # drift in a contributing model makes the router more cautious
            risk = min(1.0, risk + self.config.drift_risk_bump)
        return risk

    def route(self, ctx: DecisionContext, *, stateless: bool = False,
              needs_tool_data: bool = False) -> RoutingResult:
        cfg = self.config
        risk = self.effective_risk(ctx)

        # --- protocol selection ---
        if stateless and ctx.complexity <= 1:
            protocol = Protocol.HTTP
            proto_reason = "stateless single lookup"
        elif ctx.complexity < cfg.complexity_a2a_min and needs_tool_data:
            protocol = Protocol.MCP
            proto_reason = "single-source tool/data access"
        else:
            protocol = Protocol.A2A
            proto_reason = f"multi-agent ({ctx.complexity} parties)"

        # --- governance selection ---
        if ctx.cost_usd >= cfg.cost_human_approval_usd:
            governance = GovernanceMode.HUMAN_APPROVAL
            gov_reason = f"cost ${ctx.cost_usd:,.0f} >= human gate ${cfg.cost_human_approval_usd:,.0f}"
        elif ctx.cost_usd >= cfg.cost_escalation_usd or risk >= cfg.risk_escalation:
            governance = GovernanceMode.CENTRALIZED
            gov_reason = (
                f"cost ${ctx.cost_usd:,.0f} >= ${cfg.cost_escalation_usd:,.0f}"
                if ctx.cost_usd >= cfg.cost_escalation_usd
                else f"risk {risk:.2f} >= {cfg.risk_escalation:.2f}"
                + (" (incl. drift bump)" if ctx.model_drift_flag else "")
            )
        else:
            governance = GovernanceMode.DECENTRALIZED
            gov_reason = f"cost ${ctx.cost_usd:,.0f} and risk {risk:.2f} below thresholds"

        return RoutingResult(
            protocol=protocol,
            governance=governance,
            reason=f"protocol: {proto_reason}; governance: {gov_reason}",
        )
