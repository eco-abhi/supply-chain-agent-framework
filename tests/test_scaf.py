"""Tests for SCAF core components. Run: python3 -m pytest tests/ -q"""
import pytest

from scaf import (
    Chain, DecisionContext, DriftMonitor, GovernanceMode, GovernanceRouter,
    GuardrailEngine, GuardrailViolation, Protocol, RouterConfig,
)


# ---------- router ----------

def test_routine_decision_stays_decentralized():
    r = GovernanceRouter()
    ctx = DecisionContext(cost_usd=5_000, risk_score=0.2, complexity=2)
    out = r.route(ctx)
    assert out.governance == GovernanceMode.DECENTRALIZED
    assert out.protocol == Protocol.A2A


def test_cost_threshold_escalates():
    r = GovernanceRouter()
    ctx = DecisionContext(cost_usd=30_000, risk_score=0.2, complexity=2)
    assert r.route(ctx).governance == GovernanceMode.CENTRALIZED


def test_risk_threshold_escalates():
    r = GovernanceRouter()
    ctx = DecisionContext(cost_usd=5_000, risk_score=0.75, complexity=2)
    assert r.route(ctx).governance == GovernanceMode.CENTRALIZED


def test_human_gate():
    r = GovernanceRouter()
    ctx = DecisionContext(cost_usd=150_000, risk_score=0.1, complexity=2)
    assert r.route(ctx).governance == GovernanceMode.HUMAN_APPROVAL


def test_drift_bump_can_tip_escalation():
    r = GovernanceRouter()
    base = DecisionContext(cost_usd=5_000, risk_score=0.5, complexity=2)
    assert r.route(base).governance == GovernanceMode.DECENTRALIZED
    drifted = DecisionContext(cost_usd=5_000, risk_score=0.5, complexity=2,
                              model_drift_flag=True)
    assert r.route(drifted).governance == GovernanceMode.CENTRALIZED


def test_protocol_selection():
    r = GovernanceRouter()
    stateless = DecisionContext(cost_usd=0, complexity=1)
    assert r.route(stateless, stateless=True).protocol == Protocol.HTTP
    lookup = DecisionContext(cost_usd=0, complexity=1)
    assert r.route(lookup, needs_tool_data=True).protocol == Protocol.MCP
    multi = DecisionContext(cost_usd=0, complexity=3)
    assert r.route(multi).protocol == Protocol.A2A


def test_custom_thresholds():
    r = GovernanceRouter(RouterConfig(cost_escalation_usd=1_000))
    ctx = DecisionContext(cost_usd=2_000, risk_score=0.1, complexity=2)
    assert r.route(ctx).governance == GovernanceMode.CENTRALIZED


# ---------- guardrails ----------

def test_po_ceiling_enforced():
    g = GuardrailEngine()
    with pytest.raises(GuardrailViolation):
        g.check_po_approval("finance", 500_000)
    g.check_po_approval("finance", 200_000)  # under ceiling: no raise


def test_non_finance_cannot_approve():
    g = GuardrailEngine()
    with pytest.raises(GuardrailViolation):
        g.check_po_approval("demand", 1)


def test_tool_scoping():
    g = GuardrailEngine()
    g.check_tool_access("demand", "get_inventory")
    with pytest.raises(GuardrailViolation):
        g.check_tool_access("demand", "approve_po")
    g.check_tool_access("orchestrator", "approve_po")  # wildcard


def test_a2a_message_validation():
    g = GuardrailEngine()
    g.validate_a2a_message("supplier_risk", {"risk": 0.4})
    with pytest.raises(GuardrailViolation):
        g.validate_a2a_message("supplier_risk", {"risk": 3.5})
    with pytest.raises(GuardrailViolation):
        g.validate_a2a_message("finance", {"amount_usd": -50})


# ---------- drift ----------

def test_psi_detects_shift():
    ref = [0.3] * 50 + [0.31] * 50
    m = DriftMonitor("test", reference=ref)
    for _ in range(100):  # exactly one full batch
        m.observe(0.8)
    assert m.drifting


def test_no_drift_on_stable_data():
    import random
    rng = random.Random(0)
    ref = [rng.gauss(0.3, 0.05) for _ in range(200)]
    m = DriftMonitor("test", reference=ref)
    for _ in range(100):
        m.observe(rng.gauss(0.3, 0.05))
    assert not m.drifting


def test_disagreement_trigger():
    m = DriftMonitor("test", reference=[0.3] * 100)
    for _ in range(20):
        m.record_negotiation(agreed=False)
    assert m.drifting


def test_retrain_resets():
    m = DriftMonitor("test", reference=[0.3] * 100)
    for _ in range(100):
        m.observe(0.9)
    assert m.drifting
    m.retrain()
    assert not m.drifting
    assert m.retrain_count == 1


# ---------- scql ----------

def test_scql_cheaper_than_json():
    ch = Chain("t")
    ch.define("P1", "Part", id=4471, name="Widget")
    ch.define("S1", "Supplier", id="X1", name="Acme")
    ch.emit("MCP", "get", "P1.stock")
    ch.emit("A2A", "ask", "supplier_risk", "S1 risk?")
    assert ch.scql_tokens() < ch.json_equivalent_tokens()
    # meaningful margin, not a rounding artifact
    assert ch.json_equivalent_tokens() / ch.scql_tokens() > 2
