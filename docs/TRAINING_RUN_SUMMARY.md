# Training Run Summary

> Covers all fully-documented PPO training runs (those with `gym_environment_snapshot.py`,
> `environment_spec.json`, and `run_manifest.json`). Five pre-snapshot runs from March 2026
> were deleted as they lacked reproducibility metadata and had incompatible observation spaces.
>
> **Evaluation protocol (all runs):** held-out test day `496528674@qq.com_20161128`,
> `eval_seed=42`, greedy warmup (30 min) → composite static λ baseline vs RL agent (8-hour episode).

---

## Shared Configuration (all runs)

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Stable Baselines 3) |
| Observation space | Box, 17-dim, float32, unbounded |
| Action space | Box, 2-dim: λ₁ ∈ [0, 2.0], λ₂ ∈ [0, 0.5] |
| λ₃ (utility/distance anchor) | Fixed at 1.0 |
| Step duration | 5 minutes |
| Warmup | 30 minutes (greedy) |
| Episode length | 8 hours |
| Train/test split | 24 train days / 6 test days, `random_state=42`, shuffled |
| Data | DiDi GAIA full dataset (`data/didi/full_didi_gaia`) |
| Workers | 8 CPUs, VecEnv (SubprocVecEnv) |

**PPO Hyperparameters** (same `best_hyperparameters.json` across all runs — SHA `bb834b0e`):

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 3e-4 |
| `n_steps` | 2048 |
| `batch_size` | 256 |
| `gamma` | 0.95 |
| `gae_lambda` | 0.9 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.04 |
| `vf_coef` | 0.55 |
| `max_grad_norm` | 1.0 |
| `net_arch_type` | large |

---

## Run 1 — `run_20260402_135637`

| Field | Value |
|-------|-------|
| **Date** | 2026-04-02 (backfilled 2026-04-03 01:18 UTC) |
| **Machine** | Local Mac (backfill applied after-the-fact) |
| **Timesteps** | Unknown (pre-backfill run; manifest reconstructed) |
| **Resumed from** | N/A |
| **Git commit** | N/A (backfill from local working tree) |
| **Provenance note** | ⚠️ Snapshot generated at **backfill time** from current `rl/gym_environment.py` (SHA `8091a9b0`) — not the exact code used during training. Treat with caution for reward comparisons. |

**Reward function** (gym SHA `8091a9b0`, from `gym_environment_snapshot.py` at backfill time):

- `r_fairness = fairness × 50.0`
- `r_starvation = -recent_expirations × 0.5`
- `r_latency = -latency × 2.0`
- Combined linearly with `reward_weights = [1.0, 1.0, 1.0]`
- Normalised: `(reward − 20.0) / 2.0`
- **No SLA threshold** — the original "50.0 / 2.0 absolute" linear reward

**Results vs static baseline (eval day, seed 42):**

| Checkpoint | JFI RL / Static | Δ JFI | Peak Backlog RL / Static | Δ Backlog | Avg Wait RL / Static (m) | Δ Wait |
|-----------|----------------|-------|--------------------------|-----------|--------------------------|--------|
| **Final** | 0.5461 / 0.5616 | 🔴 −0.0155 | 103 / 105 | 🟢 −2 | 2.62 / 2.67 | 🟢 −0.05 |
| **Best** | 0.5501 / 0.5652 | 🔴 −0.0151 | 103 / 103 | ➖ 0 | 2.65 / 2.66 | 🟢 −0.02 |

---

## Run 2 — `run_20260403_122313`

| Field | Value |
|-------|-------|
| **Date** | 2026-04-03 01:23 UTC |
| **Machine** | EC2 (`/home/ec2-user/sc_sim`) |
| **Timesteps** | 50,000 |
| **Resumed from** | — (fresh run) |
| **Git commit** | `0e42e355` (clean) |

**Reward function** (gym SHA `6dd94a36`):

- `r_fairness = fairness × 100.0` ← boosted from 50.0 vs Run 1
- `r_starvation = -recent_expirations × 0.5`
- `r_latency = -latency × 2.0`
- Combined linearly with `reward_weights = [1.0, 1.0, 1.0]`
- Normalised: `(reward − 50.0) / 5.0`
- **No SLA threshold** — linear latency penalty, fairness scale doubled vs Run 1

**Results vs static baseline:**

| Checkpoint | JFI RL / Static | Δ JFI | Peak Backlog RL / Static | Δ Backlog | Avg Wait RL / Static (m) | Δ Wait |
|-----------|----------------|-------|--------------------------|-----------|--------------------------|--------|
| **Final** | 0.5475 / 0.5650 | 🔴 −0.0175 | 103 / 100 | 🔴 +3 | 2.62 / 2.67 | 🟢 −0.06 |
| **Best** | 0.5509 / 0.5636 | 🔴 −0.0126 | 104 / 101 | 🔴 +3 | 2.64 / 2.67 | 🟢 −0.03 |

---

## Run 3 — `run_20260430_154704`

| Field | Value |
|-------|-------|
| **Date** | 2026-04-30 05:47 UTC |
| **Machine** | EC2 (`/home/ec2-user/sc_sim`) |
| **Timesteps** | 50,000 |
| **Resumed from** | — (fresh run) |
| **Git commit** | `52097754` (clean) |

**Reward function** (gym SHA `7ad4a4ac`):

- **Piecewise SLA reward** — first SLA run
- `r_fairness = fairness × 100.0` (absolute JFI)
- `r_starvation = -recent_expirations × 0.5`
- If `latency ≤ 3.0 min` (safe zone): no latency penalty
- If `latency > 3.0 min`: `r_latency = -20.0 × (latency − 3.0)`
- Normalised: `(reward − 50.0) / 5.0`

**Results vs static baseline:**

| Checkpoint | JFI RL / Static | Δ JFI | Peak Backlog RL / Static | Δ Backlog | Avg Wait RL / Static (m) | Δ Wait |
|-----------|----------------|-------|--------------------------|-----------|--------------------------|--------|
| **Final** | 0.5515 / 0.5636 | 🔴 −0.0121 | 101 / 100 | 🔴 +1 | 2.61 / 2.67 | 🟢 −0.05 |
| **Best** | 0.5504 / 0.5642 | 🔴 −0.0137 | 101 / 101 | ➖ 0 | 2.62 / 2.66 | 🟢 −0.04 |

---

## Run 4 — `run_20260501_135623`

| Field | Value |
|-------|-------|
| **Date** | 2026-05-01 03:56 UTC |
| **Machine** | EC2 (`/home/ec2-user/sc_sim`) |
| **Timesteps** | 50,000 |
| **Resumed from** | `run_20260501_134829/ppo_sc_interrupted.zip` (which itself resumed from `run_20260430_212232`) |
| **Git commit** | `29f945d5` (clean) |
| **Note** | This is the **third segment** of a multi-session run chain: `run_20260430_212232` → `run_20260501_134829` → this run. |

**Reward function** (gym SHA `c0a193cb`):

- **Piecewise SLA reward** — second SLA run, with a key change to fairness signal
- `r_fairness = delta_jfi × 1000.0` ← **switched from absolute JFI to step-over-step ΔJFI**
- `r_starvation = -recent_expirations × 0.5`
- If `latency ≤ 3.0 min` (safe zone): no latency penalty
- If `latency > 3.0 min`: `r_latency = -20.0 × (latency − 3.0)`
- Normalised: `reward / 5.0` (no shift — different from all prior runs)

**Results vs static baseline:**

| Checkpoint | JFI RL / Static | Δ JFI | Peak Backlog RL / Static | Δ Backlog | Avg Wait RL / Static (m) | Δ Wait |
|-----------|----------------|-------|--------------------------|-----------|--------------------------|--------|
| **Final** | 0.5515 / 0.5644 | 🔴 −0.0128 | 107 / 103 | 🔴 +4 | 2.66 / 2.68 | 🟢 −0.02 |
| **Best** | 0.5511 / 0.5697 | 🔴 −0.0185 | 103 / 103 | ➖ 0 | 2.65 / 2.66 | 🟢 −0.02 |

---

## Cross-Run Comparison (Best Checkpoints)

| Run | JFI (RL) | JFI (Static) | Δ JFI | Peak Backlog (RL) | Avg Wait (RL, m) | Reward formula |
|-----|---------|-------------|-------|------------------|-----------------|----------------|
| Run 1 (Apr 2, backfill) | 0.5501 | 0.5652 | −0.0151 | 103 | 2.65 | Linear: `JFI×50`, `latency×−2`, norm `(r−20)/2` |
| Run 2 (Apr 3) | 0.5509 | 0.5636 | −0.0126 | 104 | 2.64 | Linear: `JFI×100`, `latency×−2`, norm `(r−50)/5` |
| Run 3 (Apr 30) | 0.5504 | 0.5642 | −0.0137 | 101 | 2.62 | **SLA piecewise**: `JFI×100`, cliff at 3.0 min |
| Run 4 (May 1, resumed) | 0.5511 | 0.5697 | −0.0185 | 103 | 2.65 | **SLA piecewise**: `ΔJFI×1000`, cliff at 3.0 min |

> **Run 2 (Apr 3)** had the smallest JFI regression (−0.0126). **Run 3 (Apr 30)** produced the lowest peak backlog (101) and best avg wait (2.62 m). Neither SLA run (3 & 4) improved on these — the cliff penalty did not help fairness.

---

## Consistent Observations Across All Runs

1. **Wait time improves** (0.02–0.06 m) vs static baseline in every run — the agent does learn to dispatch faster.
2. **JFI regresses** in every run (−0.012 to −0.019) — the agent sacrifices fairness to reduce wait, the opposite of what we want.
3. **Peak backlog is comparable or slightly worse** — suggesting the agent is not causing dangerous queue build-up but isn't significantly clearing it either.
4. **Consistent hyperparameters** — same `best_hyperparameters.json` (SHA `bb834b0e`) across all four runs.
5. **Same gym env SHA** would be expected for a consistent experiment; Runs 2–4 all use slightly different gym SHAs, indicating the reward code evolved between sessions.

---

## What Changed Between Runs

| Between | Change |
|---------|--------|
| Run 1 → Run 2 | Fairness scale doubled (`×50` → `×100`); normalisation updated (`(r−20)/2` → `(r−50)/5`) |
| Run 2 → Run 3 | **Linear → SLA piecewise reward**: latency cliff at 3.0 min replaces continuous `−latency×2` penalty |
| Run 3 → Run 4 | **Absolute JFI → ΔJFI** fairness signal (`JFI×100` → `delta_jfi×1000`); normalisation shift removed (`(r−50)/5` → `r/5`) |
| Run 4 → **Next** | SLA confirmed incompatible with dataset variance; next step is Option A (300k linear) or Option B (action smoothing) |

---

## SLA Reward — Explored and Discarded

A piecewise SLA reward was implemented in `gym_environment.py` (`sla_wait_time_minutes=3.0`, `sla_violation_penalty=20.0`) and the calibration script (`scripts/calibrate_sla.py`) was written to derive a data-driven threshold.

Running `calibrate_sla.py` across 30 days revealed **why the SLA approach is incompatible with this dataset:**

- Pooled p75 step-wise wait: ~2.40 minutes
- Hard days (e.g. `20161101`) produce natural spikes up to **7.38 minutes** under the static baseline

Setting any hard threshold means the agent is punished with `−20 × violation` purely because of natural traffic conditions on a busy day, not because of its own actions. This causes the agent to collapse to `[0, 0]` (λ₁ = 0, λ₂ = 0) as a defensive local optimum — "learning" to do nothing to avoid the cliff.

**Conclusion:** The SLA reward is theoretically sound but empirically incompatible with the high day-to-day variance of the DiDi GAIA dataset. This is a useful finding in itself — reviewers who ask "why not a threshold penalty?" now have a documented, data-driven answer.

The SLA code remains in the codebase but should not be used without a per-day dynamic threshold (a much larger engineering lift).

---

## Next Planned Run (Active Decision Point)

The core problem across all 4 runs is the same: the agent **trades JFI for wait time** instead of improving both. The linear reward gives it no clear signal to prioritise fairness.

Two options are on the table — both revert to or build on the **smooth linear reward** (no SLA cliff):

### Option A — 300k Marathon (recommended first)

Re-run with the same smooth reward as Runs 2–4 but give the agent enough steps to properly explore the policy space. At 50k steps / ~96 steps per episode the agent only completes ~520 episodes — barely enough to finish exploring, let alone converge.

- **Reward:** unchanged linear (`r_fairness × 100`, `r_latency × −2`, `r_starvation × −0.5`, normalised `(reward − 50) / 5`)
- **Timesteps:** 300,000
- **Everything else:** same `best_hyperparameters.json`, same data split
- **Rationale:** The April 2nd / 3rd runs showed the agent can reach λ₁ = 0.16 before retreating. More steps = more time to precisely map the fairness ceiling and learn to stay there.

### Option B — Action Smoothing Penalty

Keep the 50k–60k format but add a **smoothness penalty** on large step-to-step action changes. The agent currently jumps (e.g. λ₁: 0.04 → 0.55 in one step), crashing the spatial equilibrium and then retreating to near-zero.

A penalty on `|action[t] − action[t−1]|` forces gradual dial-turns (0.04 → 0.08 → 0.12), giving the city's physics time to re-equilibrate before the next change.

- Requires a small code change in `_calculate_reward` (store `self.last_action`, penalise delta)
- Lower engineering cost than a dynamic SLA
- Can be combined with a 300k run later if Option A alone stalls

**Command (either option):**

```bash
python rl/train_sb3.py --timesteps 300000 --hyperparams best_hyperparameters.json
```

**Success criteria:** Best checkpoint JFI ≥ static baseline JFI (~0.56+), avg wait within 0.1 m of baseline.
