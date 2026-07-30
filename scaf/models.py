"""Core data models for the SCAF framework.

Decision contexts and the enums the router selects between.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class Protocol(Enum):
    HTTP = "http"   # stateless lookup, no context needed
    MCP = "mcp"     # agent -> tool/data access
    A2A = "a2a"     # agent -> agent negotiation/delegation


class GovernanceMode(Enum):
    DECENTRALIZED = "decentralized"   # peer A2A negotiation, autonomous
    CENTRALIZED = "centralized"       # routed through orchestrator
    HUMAN_APPROVAL = "human_approval" # centralized + human sign-off gate


@dataclass
class DecisionContext:
    """Everything the governance router needs to route one decision."""
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = ""
    cost_usd: float = 0.0            # monetary size of the decision
    risk_score: float = 0.0          # 0..1, from supplier risk agent + model confidence
    complexity: int = 1              # number of agents/data sources needed
    model_drift_flag: bool = False   # a contributing model is drifting
    created_at: float = field(default_factory=time.time)


@dataclass
class RoutingResult:
    protocol: Protocol
    governance: GovernanceMode
    reason: str


@dataclass
class AuditRecord:
    """One entry in the audit log: who, what, why. Feeds instant reporting."""
    decision_id: str
    actor: str
    action: str
    rationale: str
    protocol: Protocol | None = None
    governance: GovernanceMode | None = None
    timestamp: float = field(default_factory=time.time)
