"""SCAF: Supply Chain Agent Framework.

Reference implementation of the governed orchestration framework:
dynamic protocol + governance-mode selection for multi-agent
supply chain systems.
"""
from .models import DecisionContext, GovernanceMode, Protocol
from .router import GovernanceRouter, RouterConfig
from .guardrails import GuardrailEngine, GuardrailViolation, Policy
from .drift import DriftMonitor, psi

__all__ = [
    "DecisionContext", "GovernanceMode", "Protocol",
    "GovernanceRouter", "RouterConfig",
    "GuardrailEngine", "GuardrailViolation", "Policy",
    "DriftMonitor", "psi",
]
__version__ = "0.1.0"
