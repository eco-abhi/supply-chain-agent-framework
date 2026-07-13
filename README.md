# SCAF: Supply Chain Agent Framework

A reference implementation of a **governed orchestration framework** for
autonomous supply chain systems. The core idea: the choice of communication
protocol (HTTP / MCP / A2A) and the choice of governance mode (centralized
vs decentralized) shouldn't be fixed at design time. A runtime router should
make both choices together, per decision, based on the risk and cost of that
specific decision.

Companion code for the paper *"A Governed Orchestration Framework for
Autonomous Supply Chains: Dynamic Protocol and Governance-Mode Selection
for Multi-Agent Systems."*

## The components

| Component | Module | What it does |
|---|---|---|
| Governance router | `scaf/router.py` | Picks (protocol, governance mode) per decision from risk, cost, complexity, and drift signals |
| Guardrails | `scaf/guardrails.py` | Policy-as-code: hard PO ceilings, per-agent tool scoping, A2A message validation, mandatory audit logging |
| Drift monitor | `scaf/drift.py` | PSI-based drift detection plus negotiation-disagreement as a second retraining trigger |
| SCQL notation | `scaf/scql.py` | Compact cross-protocol query notation: define entities once, reference by alias, cut token cost |
| Multi-party header caching | `scaf/multiparty.py` | Self-describing headers sent once per participant; a `RemoteParticipant` decodes wire messages with no shared memory |
| Agents | `scaf/agents/` | Demand, supplier risk, finance agents plus the central orchestrator |
| Evaluation experiments | `scaf/experiments.py` | Threshold sensitivity, learned-baseline comparison, adversarial injection, retraining ablation |

## Quick start

```bash
# run the worked example (stockout scenario, three governance paths)
python3 examples/demo.py

# run the three-way benchmark (centralized vs decentralized vs dynamic)
python3 -m scaf.benchmark

# run the paper's additional evaluation experiments (Section 5.2-5.5)
python3 -m scaf.experiments

# run the multi-party header-caching benchmark (Section 5.6)
python3 -m scaf.multiparty

# run tests
python3 -m pytest tests/ -q
```

## How routing works

```python
from scaf import GovernanceRouter, DecisionContext

router = GovernanceRouter()

# routine reorder: cheap, low risk -> decentralized A2A, no escalation
ctx = DecisionContext(cost_usd=12_000, risk_score=0.2, complexity=2)
router.route(ctx).governance  # GovernanceMode.DECENTRALIZED

# same decision but the risk model is drifting -> router gets cautious
ctx = DecisionContext(cost_usd=12_000, risk_score=0.5, complexity=2,
                      model_drift_flag=True)
router.route(ctx).governance  # GovernanceMode.CENTRALIZED

# big order -> human sign-off required
ctx = DecisionContext(cost_usd=150_000, risk_score=0.2, complexity=2)
router.route(ctx).governance  # GovernanceMode.HUMAN_APPROVAL
```

## Benchmark results (1,000 simulated decisions)

| Config | Mean latency | Governance coverage | Autonomy | Token cost |
|---|---|---|---|---|
| Static centralized | 7.04 | 100% | 0% | baseline |
| Static decentralized | 1.00 | 0% | 100% | baseline |
| **Dynamic router** | **5.54** | **100%** | **100%** | baseline |
| SCQL vs verbose JSON | | | | **~5.6x fewer tokens** |

Governance coverage = share of high-stakes decisions (cost >= $25k or
risk >= 0.6) that received centralized oversight. Autonomy = share of
routine decisions that resolved without escalation. Only the dynamic
router achieves both at once. Latency units are simulated and relative.

## Status and caveats

This is a reference implementation, not production software. The agents
use stubbed models and simulated data. The latency numbers are relative
simulation units. Plugging in a real training pipeline (`DriftMonitor.retrain`)
and real transport (actual MCP servers and A2A agent cards) is the natural
next step.

## Citation

Companion code for:

> Pandey, A. *A Governed Orchestration Framework for Autonomous Supply
> Chains: Dynamic Protocol and Governance-Mode Selection for Multi-Agent
> Systems.* Submitted to the *International Journal of Production
> Research*, Special Issue: The Agentic Supply Chain (under review).

A full citation with volume/issue/DOI will be added once the paper is
published.

## License

Apache License 2.0. See [LICENSE](LICENSE).

