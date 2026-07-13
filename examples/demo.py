"""Runnable demo: the paper's worked example, three ways.

  1. routine order, low risk        -> decentralized, autonomous
  2. same order, drifting model     -> escalates to central orchestrator
  3. big order                      -> human sign-off gate
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scaf.scenario import run_stockout_scenario  # noqa: E402

print("=" * 60)
print("Scenario 1: routine order, low risk, healthy model")
print("=" * 60)
run_stockout_scenario(seed=1, order_cost=12_000)

print()
print("=" * 60)
print("Scenario 2: same order, but the risk model is drifting")
print("=" * 60)
run_stockout_scenario(seed=1, order_cost=12_000, inject_drift=0.35)

print()
print("=" * 60)
print("Scenario 3: big order crosses the human-approval gate")
print("=" * 60)
run_stockout_scenario(seed=1, order_cost=150_000)
