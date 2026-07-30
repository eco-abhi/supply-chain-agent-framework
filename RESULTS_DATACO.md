# DataCo replication results (side-by-side with Olist)

Spine-only replication on DataCo Smart Supply Chain. Same methodology as
Olist Exp1 (70/30 time split, LightGBM config, rank-blend dynamic score,
bootstrap CIs, DeLong, `coverage_adverse` sweep). No threshold-sensitivity,
retraining, adversarial, or protocol-distribution runs (single-supplier).

- DataCo n=180,519 item-rows; train=126,363 / test=54,156
  (cut at 2017-01-08).
- Label: `adverse = (days_real > days_scheduled) OR Shipping canceled`
  (`Late_delivery_risk` unused). Cost = Order Item Total.

## Base rate & risk model

| Metric | DataCo | Olist |
|---|---:|---:|
| Test base rate | **0.5899** | 0.0635 |
| Risk model AUC | **0.734** | 0.719 |
| Risk model AP | 0.836 | — |
| Risk model Brier | 0.1953 | — |

Note: DataCo adverse base rate ≈ **0.59** (near 0.5 as expected;
overall 0.59). Olist test base rate is ~6%.

## Ranker AUC of `adverse_outcome` (test)

| Config | DataCo AUC [95% CI] | Olist AUC [95% CI] |
|---|---|---|
| Cost-only | **0.501** [0.496, 0.505] | 0.497 [0.482, 0.511] |
| Dynamic | **0.668** [0.663, 0.673] | 0.633 [0.620, 0.645] |
| Risk-only | **0.734** [0.730, 0.738] | 0.719 [0.709, 0.730] |

### DeLong (dynamic − cost-only)

| | DataCo | Olist |
|---|---|---|
| ΔAUC | **+0.1675** | +0.136 |
| z | **86.704** | 20.0 |
| p | **≈ 0 (underflow)** | ≈ 5e-89 |

## `coverage_adverse` Δ (dynamic − cost) at matched escalation

| Escalation | DataCo Δ | Olist Δ |
|---:|---:|---:|
| 10% | **+0.037** | +0.068 |
| 15% | **+0.052** | +0.085 |
| 20% | **+0.066** | +0.124 |
| 25% | **+0.082** | +0.147 |
| 30% | **+0.095** | +0.180 |

### Absolute coverage_adverse (DataCo)

| Escalation | Dynamic | Cost-only | Δ |
|---:|---:|---:|---:|
| 10% | 0.089 | 0.052 | +0.037 |
| 15% | 0.158 | 0.105 | +0.052 |
| 20% | 0.225 | 0.159 | +0.066 |
| 25% | 0.295 | 0.212 | +0.082 |
| 30% | 0.359 | 0.264 | +0.095 |

## Artifacts

- `results/dataco_replication.csv`
- `figures/dataco_coverage_adverse.png`
