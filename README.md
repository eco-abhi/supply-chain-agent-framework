# SCAF: Supply Chain Agent Framework

Companion code for *"A Governed Orchestration Framework for Autonomous
Supply Chains: Dynamic Protocol and Governance-Mode Selection for
Multi-Agent Systems"* (IJPE, under review).

A runtime **governance router** chooses protocol (HTTP / MCP / A2A) and
governance mode (decentralized / centralized / human approval) per
decision from cost, risk, complexity, and drift — rather than fixing
those choices at design time.

The paper evaluation uses real order data (Olist, DataCo). The `scaf/`
package is a small reference implementation of the router, guardrails,
and agents used in the worked example.

## Paper evaluation (primary)

| Piece | Path | What it does |
|---|---|---|
| Olist pipeline (§5) | `src/` + `run_all.py` | Risk model, rankers, frontier, learned baseline, adversarial, retraining |
| Protocol shares (§5.4b) | `src/experiments/protocol_distribution.py` | HTTP / MCP / A2A on the locked test split |
| DataCo replication (§5.9) | `src/dataco/` | Spine only: risk + rankers + `coverage_adverse` |
| Expected cost (§5.9) | `src/experiments/expected_cost.py` | Cost-only vs dynamic min expected cost vs ρ |

Narrative numbers: [`RESULTS.md`](RESULTS.md) (Olist) and
[`RESULTS_DATACO.md`](RESULTS_DATACO.md) (DataCo side-by-side).
CSV/figures land in `results/` and `figures/` (gitignored).

### Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Datasets (CSVs are not in git): see [`data/README.md`](data/README.md)
and [`data/dataco/README.md`](data/dataco/README.md).

### Run

```bash
python -m src.risk_model                       # locked Olist risk model
python run_all.py                              # exp1–exp5
python -m src.experiments.protocol_distribution
python -m src.dataco.replicate                 # DataCo spine
python -m src.experiments.expected_cost        # §5.9 expected-cost curves
```

## Reference framework (`scaf/`)

| Component | Module | Role |
|---|---|---|
| Governance router | `scaf/router.py` | Protocol + governance mode from cost, risk, complexity, drift |
| Guardrails | `scaf/guardrails.py` | PO ceilings, tool scoping, A2A validation, audit log |
| Drift monitor | `scaf/drift.py` | PSI + negotiation-disagreement triggers |
| Agents | `scaf/agents/` | Demand, supplier risk, finance, orchestrator (stubs) |
| Synthetic demos | `scaf/benchmark.py`, `scaf/experiments.py` | Toy latency / coverage checks (not the paper tables) |

```bash
python examples/demo.py          # worked example, three governance paths
python -m scaf.benchmark         # synthetic centralized vs decentralized vs dynamic
python -m pytest tests/ -q
```

### How routing works

```python
from scaf import GovernanceRouter, DecisionContext

router = GovernanceRouter()

# routine reorder: cheap, low risk -> decentralized A2A
ctx = DecisionContext(cost_usd=12_000, risk_score=0.2, complexity=2)
router.route(ctx).governance  # GovernanceMode.DECENTRALIZED

# same decision, drifting risk model -> escalate
ctx = DecisionContext(cost_usd=12_000, risk_score=0.5, complexity=2,
                      model_drift_flag=True)
router.route(ctx).governance  # GovernanceMode.CENTRALIZED

# large order -> human sign-off
ctx = DecisionContext(cost_usd=150_000, risk_score=0.2, complexity=2)
router.route(ctx).governance  # GovernanceMode.HUMAN_APPROVAL
```

## Citation

> Pandey, A. *A Governed Orchestration Framework for Autonomous Supply
> Chains: Dynamic Protocol and Governance-Mode Selection for Multi-Agent
> Systems.* Submitted to the *International Journal of Production
> Economics*, Special Issue: The Agentic Supply Chain (under review).

A full citation with volume/issue/DOI will be added once the paper is
published.

## License

Apache License 2.0. See [LICENSE](LICENSE).
