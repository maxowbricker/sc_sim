# Oracle Architecture — 50k Run Comparison

Three new 50k runs all use a **greedy-step oracle** variant of the environment
(state is snapshotted each step, a greedy pass is run from the exact same state,
and the reward is computed as RL-vs-oracle advantage). What differs between them is
**which reference policy acts as the "oracle"** and **how the reward is shaped**.

> Evaluation protocol shared with all prior runs:  
> Test day `496528674@qq.com_20161128`, `eval_seed=42`, 30 min greedy warmup → 8 h episode.

---

## Run identification

| Run folder | Branch | Oracle baseline | Reward shape | Git commit |
|---|---|---|---|---|
| `run_20260511_190345` | `oracle-static-composite` | **Static composite** (fixed λ₁=1.0, λ₂=0.2, λ₃=1.0) | Symmetric linear advantage; fairness ×**1000** | `8d36972` |
| `run_20260511_230040` | `oracle-dynamic-sla` (revised) | **Greedy oracle** | Capped Dynamic SLA; fairness ×**1000**, penalty floor −10.0 | `475a788` |
| `run_20260512_041841` | `oracle-approach` (revised) | **Greedy oracle** | Asymmetric linear; fairness ×**1000**, latency/starvation capped at 0 | `02c3792` |

All three ran on EC2 with `--timesteps 50000 --hyperparams best_hyperparameters.json`,
clean repo (`dirty=false`), same hyperparams SHA (`bb834b0e`).

---

## Reward functions

### `run_20260511_190345` — Symmetric Static-Composite Advantage

```python
composite_stats = self.simulator.metrics.get_reward_stats(...)
oracle_stats    = self.oracle_reward_stats   # static composite run from same state

fairness_adv   = composite_stats['fairness']           - oracle_stats['fairness']
r_fairness     = fairness_adv * 1000.0                 # 10× boost vs prior oracle runs

latency_adv    = oracle_stats['latency']               - composite_stats['latency']
r_latency      = latency_adv * 10.0                    # symmetric: reward for being faster

starvation_adv = oracle_stats['recent_expirations']    - composite_stats['recent_expirations']
r_starvation   = starvation_adv * 1.0

return (r_fairness + r_latency + r_starvation) / 5.0
```

Key difference: the oracle is **the agent's own fixed-weight competitor**, not greedy.
This means "neutral reward" = exactly matching the static composite, which is the
paper's primary comparison target.

---

### `run_20260511_230040` — Capped Dynamic SLA (revised oracle-dynamic-sla)

```python
fairness_adv   = composite_stats['fairness'] - oracle_stats['fairness']
r_fairness     = fairness_adv * 1000.0                 # 10× vs prior oracle-dynamic-sla

latency_diff   = composite_stats['latency'] - oracle_stats['latency']
if latency_diff <= 0.1:                                # 6-second safe zone (unchanged)
    r_latency = 0.0
else:
    r_latency = max(-10.0, -10.0 * latency_diff)      # NEW: penalty capped at −10.0

starvation_adv = oracle_stats['recent_expirations'] - composite_stats['recent_expirations']
r_starvation   = starvation_adv * 1.0

return (r_fairness + r_latency + r_starvation) / 5.0
```

Prior version (`run_20260510_133620`): same gate shape but fairness ×100, uncapped penalty.
The cap was added to prevent extreme latency spikes from dominating the signal.

---

### `run_20260512_041841` — Asymmetric Linear Advantage (revised oracle-approach)

```python
fairness_adv   = composite_stats['fairness'] - oracle_stats['fairness']
r_fairness     = fairness_adv * 1000.0                 # 10× vs prior oracle-approach

latency_adv    = oracle_stats['latency'] - composite_stats['latency']
r_latency      = min(0.0, latency_adv) * 5.0           # no reward for beating oracle latency

starvation_adv = oracle_stats['recent_expirations'] - composite_stats['recent_expirations']
r_starvation   = min(0.0, starvation_adv) * 1.0        # no reward for fewer expirations

return (r_fairness + r_latency + r_starvation) / 5.0
```

Prior version (`run_20260510_062142`): symmetric (both directions rewarded) with fairness ×100, latency ×5.
The asymmetric version removes positive latency/starvation reward to make fairness the unambiguous goal.

---

## Performance vs static composite baseline

### Final checkpoint

| Run | JFI static | JFI RL | **ΔJFI** | BL static | BL RL | ΔBL | Wait static (m) | Wait RL (m) | ΔWait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_20260511_190345` (static-composite oracle) | 0.5692 | 0.5633 | **−0.0059** ✅ | 102 | 101 | −1 | 2.67 | 2.68 | +0.01 |
| `run_20260511_230040` (dyn-SLA oracle, revised) | 0.5658 | 0.5500 | −0.0158 | 104 | 103 | −1 | 2.68 | 2.63 | −0.05 |
| `run_20260512_041841` (asymmetric oracle) | 0.5635 | 0.5516 | −0.0119 | 103 | 103 | 0 | 2.67 | 2.64 | −0.03 |

### Best-eval-reward checkpoint

| Run | JFI static | JFI RL | **ΔJFI** | BL static | BL RL | ΔBL | Wait static (m) | Wait RL (m) | ΔWait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_20260511_190345` (static-composite oracle) | 0.5698 | 0.5675 | **−0.0023** ✅✅ | 102 | 101 | −1 | 2.68 | 2.67 | −0.01 |
| `run_20260511_230040` (dyn-SLA oracle, revised) | 0.5673 | 0.5476 | −0.0197 | 102 | 100 | −2 | 2.67 | 2.62 | −0.05 |
| `run_20260512_041841` (asymmetric oracle) | 0.5634 | 0.5501 | −0.0133 | 99 | 106 | **+7** ⚠️ | 2.66 | 2.64 | −0.02 |

---

## Comparison to closest prior runs

| New run | Most related prior run | Key change | ΔJFI then | ΔJFI now |
|---|---|---|---:|---:|
| `run_20260511_190345` | `run_20260510_062142` (oracle-approach, sym) | Oracle changed: greedy → static composite; fairness ×100 → ×1000 | −0.0136 (best) | **−0.0023 (best)** |
| `run_20260511_230040` | `run_20260510_133620` (oracle-dyn-SLA) | Fairness ×100 → ×1000; added penalty cap at −10.0 | −0.0093 (best) | −0.0197 (best) |
| `run_20260512_041841` | `run_20260510_062142` (oracle-approach, sym) | Latency/starvation reward capped at 0; fairness ×100 → ×1000 | −0.0136 (best) | −0.0133 (best) |

---

## λ-weight traces (96 eval steps, best and final policies)

λ₁ is identically 0 in all six traces. All action lives in **λ₂ (starvation weight)**.

### Final policy

| Run | λ₂ min | λ₂ max | λ₂ μ | λ₂ σ | Pattern |
|---|---:|---:|---:|---:|---|
| `run_20260511_190345` (static-composite oracle) | 0.000 | 0.192 | ~0.061 | ~0.065 | Silent steps 1–12, active band steps 13–71 (0.04–0.19), zeroes out end of day |
| `run_20260511_230040` (dyn-SLA oracle, revised) | 0.000 | 0.159 | ~0.082 | ~0.052 | Sustained high (0.09–0.16) first half, gradual decay, collapses from step 72 |
| `run_20260512_041841` (asymmetric oracle) | 0.000 | 0.036 | ~0.001 | ~0.004 | **Near-complete collapse** — almost all steps output λ₂ ≈ 0 |

### Best checkpoint policy

| Run | λ₂ min | λ₂ max | λ₂ μ | λ₂ σ | Pattern |
|---|---:|---:|---:|---:|---|
| `run_20260511_190345` (static-composite oracle) | 0.000 | 0.158 | ~0.033 | ~0.030 | Initial burst (0.158), then small but persistent activity (~0.01–0.06) across episode |
| `run_20260511_230040` (dyn-SLA oracle, revised) | 0.016 | 0.198 | ~0.087 | ~0.048 | **All 96 steps non-zero** — sustained high-to-moderate activity throughout |
| `run_20260512_041841` (asymmetric oracle) | 0.028 | 0.080 | ~0.049 | ~0.011 | **All 96 steps non-zero** — remarkably stable band (~0.04–0.07), most consistent λ₂ shape seen |

---

## Key takeaways

### 1. `run_20260511_190345` breaks the fairness regression barrier

Every prior 50k run showed ΔJFI in the range −0.009 to −0.019 (best checkpoints). This run produces
**ΔJFI = −0.0023** on the best checkpoint — the smallest fairness loss by a factor of ~4× versus the
previous record (`run_20260510_133620`, −0.0093). Wait time stays neutral (−0.01 min).

The most plausible explanation: using the **static composite as the oracle** makes the reward signal
directly track "am I beating the fixed-weight baseline?" — the exact question the paper asks. A greedy
oracle is an easier target that doesn't correspond to the paper's evaluation metric; a static-composite
oracle forces the agent to learn where the fixed λ heuristic underperforms and exploit those moments.

### 2. Boosting fairness to ×1000 helps — but only with the right oracle

All three new runs use fairness ×1000 (up from ×100). But only the static-composite oracle run
benefits from it on JFI. The two greedy-oracle runs (`run_20260511_230040`, `run_20260512_041841`)
do **not** improve on their respective predecessors on ΔJFI, and `run_20260511_230040` gets worse
(−0.0197 vs −0.0093 prior). The ×1000 scale alone is not the lever — the oracle choice is.

### 3. Asymmetric latency cap produces the most stable λ₂ policy ever seen

`run_20260512_041841` (asymmetric oracle, best ckpt) holds a steady λ₂ ∈ [0.028, 0.080] across
all 96 eval steps with σ ≈ 0.011. This is the most consistent action behaviour recorded across any
run to date, suggesting the "no reward for beating oracle on secondary objectives" constraint acts as
an implicit regulariser on the policy. The JFI result is mid-table (−0.0133) but the behavioural
consistency is interesting if combined with the better oracle choice.

### 4. `run_20260511_230040` best model's +7 peak backlog is a red flag

The best checkpoint of the asymmetric oracle run pushed peak backlog from 99 to 106 (+7). This is
the largest backlog regression across any of the 10 documented runs. The policy's stable λ₂ band
(~0.04–0.07 starvation weight) may be systematically deferring difficult tasks and degrading queue
depth. Worth monitoring if this run type is extended to longer training.

---

## Suggested next step

**Priority**: extend `run_20260511_190345` (static-composite oracle, ×1000 fairness) to a longer run
(150k–300k timesteps). At 50k it is already within 0.002 JFI of the static baseline, a result no
prior run achieved at any budget. More steps could close that gap entirely or even produce the first
positive ΔJFI.

**Ablation to consider**: rerun the asymmetric oracle (`oracle-approach` revised) but with the
static-composite oracle instead of greedy, to test whether the improved oracle choice explains
`run_20260511_190345`'s result on its own, or whether the symmetric reward term also matters.

---

## Provenance

Artifacts read for this document:

- `run_manifest.json` — argv, git commit, training params
- `gym_environment_snapshot.py` — `_calculate_reward` excerpts
- `baseline_final_model_metrics.txt`, `baseline_best_model_metrics.txt`
- `baseline_final_model_weight_outputs.txt`, `baseline_best_model_weight_outputs.txt`
