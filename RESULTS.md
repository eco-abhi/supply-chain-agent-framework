# Evaluation Results (Olist)

Companion numbers for Section 5 of the paper. Locked risk model / labels /
features / 70–30 time split are unchanged from the risk-model checkpoint.

## Operationalization assumptions (lift into Methods)

| Item | Choice |
|---|---|
| Unit of decision | One Olist **order** (auto-fulfill vs escalate) |
| Primary label `adverse_outcome` | Late delivery **or** `canceled`/`unavailable` (reviews excluded) |
| Labelled set | 96,945 orders after dropping 1,721 in-transit |
| Split | Time-ordered 70/30 at ~2018-04-15 (train 67,861 / test 29,084) |
| Validation | Final **15%** of train (10,180 orders) — **all thresholds set here only** |
| `risk_score` | Locked LightGBM P(adverse); test AUC **0.719** |
| `cost` | Σ(price + freight) in BRL |
| `high_stakes` | `adverse_outcome==1` **OR** `cost ≥` test 95th pctile (R$460.87) |
| Protocol | 1 item/1 seller→HTTP; multi-item/1 seller→MCP; multi-seller→A2A |
| Latency | **Assumed cost proxy**, not measured time: A2A=1.0, +2.5 orchestrator, +20 human |
| Dynamic score (curves) | `max(pct_rank_cost, pct_rank_risk_eff)` with drift bump 0.2 on risk |
| Exp1 three-way | `fixed_hybrid`=cost-only; `risk_only`=raw `risk_score`; `dynamic_router`=both |

---

## Section 5.3 — Main benchmark / matched escalation [[FILL: sec5.3]]

### Ranker AUC / AP of `adverse_outcome` (test) — **headline**

| Config | AUC [95% CI] | AP [95% CI] |
|---|---|---|
| Cost-only (`fixed_hybrid`) | **0.497 [0.482, 0.511]** | 0.065 [0.061, 0.070] |
| Dynamic router | **0.633 [0.620, 0.645]** | 0.099 [0.092, 0.107] |
| Risk-only | **0.719 [0.709, 0.730]** | 0.137 [0.128, 0.148] |

- DeLong (dynamic − cost): **ΔAUC = +0.136, z = 20.0, p ≈ 5×10⁻⁸⁹**
- Cost is chance-level for operational risk; risk-aware ranking recovers discrimination.

### `coverage_adverse` Δ (dynamic − cost) at matched escalation

| Escalation | Dynamic | Cost-only | Δ |
|---:|---:|---:|---:|
| 10% | 0.123 | 0.056 | **+0.068** |
| 15% | 0.197 | 0.112 | **+0.085** |
| 20% | 0.286 | 0.162 | **+0.124** |
| 25% | 0.359 | 0.212 | **+0.147** |
| 30% | 0.442 | 0.261 | **+0.180** |

Risk-only `coverage_cost` at ~50% escalation: **0.587** (still far below cost-only on that axis; see three-panel figure).

### Operating-point table (val q_cost=q_risk=0.80)

See `results/exp1.csv`. Dynamic coverage 0.802 / autonomy 0.619 / esc 0.428; cost-only coverage 0.542 / autonomy 0.845 / esc 0.198.

**Figures:** `figures/coverage_three_panel.png`, `coverage_vs_escalation.png`, `coverage_adverse_vs_escalation.png`.

---

## Section 5.4 — Threshold frontier [[FILL: sec5.4 / exp2]]

- Grid: 10×10 validation quantiles ∈ [0.50, 0.95] → 100 points; **39** on Pareto frontier.
- Exp1 operating point (q=0.80/0.80): coverage **0.802**, autonomy **0.619**, esc **0.428** — **on the frontier** (`True`).
- Figure: `figures/frontier.png`. CSV: `results/exp2.csv`.

---

## Section 5.5 — Learned baseline [[FILL: sec5.5 / exp3]]

Oracle = dual-axis `high_stakes` using **validation** cost p95 (no test leakage). Features: `[cost, risk, complexity, drift]` ± `cost×risk`.

| Config | AUC | AP | Coverage | Esc rate |
|---|---:|---:|---:|---:|
| learned_logistic | 0.803 | 0.525 | 0.471 | 0.090 |
| **learned_lgbm** | **0.841** | **0.625** | 0.498 | 0.093 |
| learned_logistic + interaction | 0.804 | 0.531 | 0.513 | 0.101 |
| learned_lgbm + interaction | 0.841 | 0.625 | 0.495 | 0.092 |
| dynamic_router | 0.707 | 0.181 | **0.802** | 0.428 |
| dynamic_router + interaction gate | 0.707 | 0.181 | 0.802 | 0.428 |

- Learned LGBM beats the threshold router on the **oracle-classification** AUC (+0.135) at far lower escalation.
- Engineered `cost×risk` interaction **does not close** the gap (router AUC unchanged 0.707; LGBM interaction ≈ identical to LGBM).
- Disagreement mass concentrates in hi-cost/hi-risk (4332) and lo-cost/hi-risk (3503) quadrants (`results/exp3.csv`).

---

## Section 5.6 — Adversarial robustness [[FILL: sec5.6 / exp4]]

Permutation attack on test `risk_score` (30 seeds). Disagreement low-risk band = validation risk ≤ q0.30 (`t_low=0.0538`).

| Monitor | Detections / 30 | Rate | Wilson 95% CI |
|---|---:|---:|---|
| PSI (>0.2 vs val reference) | **30/30** | 1.000 | [0.886, 1.000] |
| PSI baseline FPR (bootstrap) | **30/30** | 1.000 | [0.886, 1.000] |
| Disagreement (excess adverse vs mean reported) | **30/30** | 1.000 | [0.886, 1.000] |
| Disagreement FPR (joint bootstrap) | **0/30** | 0.000 | [0.000, 0.114] |

- Honest disagreement flag: **False** (realized 0.022 vs expected 0.032).
- Two-proportion (disagreement attack vs FPR): **z = 7.75, p ≈ 9.5×10⁻¹⁵**
- Two-proportion (PSI attack vs FPR): **z = 0, p = 1** — PSI rate equals its FPR; **no attack-specific lift** (marginal preserved under permutation). Real train→test drift already pushes PSI(test,val)=0.306 > 0.2.
- CSV: `results/exp4.csv`.

---

## Section 5.7 — Retraining ablation [[FILL: sec5.7 / exp5]]

Real base-rate drift (train 9.5% → test 6.35%). Calendar every 2000 decisions vs PSI>0.2 every 500-decision batch. Optional synthetic shift: multiply `customer_seller_distance_km` by 1.75 from test index 9000 (odd seeds).

| Strategy | Decisions-to-recovery | Under-governed high-risk | Mean retrains |
|---|---|---|---:|
| Calendar | **2000** (identical across 20 seeds) | **19** (identical) | 14.0 |
| PSI-triggered | **2400** (identical across 20 seeds) | **24** (identical) | 7.0 |

- All 20 seeds produced the same within-strategy totals (structural drift dominates seed noise); paired t-test is undefined (zero within-cell variance). The calendar advantage is **consistent**: −400 decisions to recovery and −5 under-governed cases vs PSI.
- Synthetic shift location/magnitude as above; did not change the within-strategy constants.
- Figure: `figures/recovery_curves.png`. CSV: `results/exp5.csv`.

---

## Artifact index

| Path | Contents |
|---|---|
| `results/risk_model_metrics.json` | Locked risk-model AUC / calibration / importances |
| `results/exp1.csv` | Operating-point table |
| `results/exp1_curves.csv` | Rank-based coverage curves (all / adverse / cost) |
| `results/exp1_ranker_metrics.csv` | AUC/AP + bootstrap CIs |
| `results/exp2.csv` | Threshold grid + frontier flag |
| `results/exp3.csv` | Learned vs router (+ interaction) |
| `results/exp4.csv` | Per-seed adversarial detections |
| `results/exp5.csv` | Per-seed recovery metrics |
| `figures/coverage_three_panel.png` | Section 5.3 three-panel |
| `figures/frontier.png` | Section 5.4 |
| `figures/recovery_curves.png` | Section 5.7 |
