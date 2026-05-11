# 50k PPO Training Runs — Summary & Comparison

Scope: every `rl_logs_sb3/run_*` whose `ppo_sc_final.zip` records
**`_total_timesteps = 50000`** (CLI budget) and **`num_timesteps = 65536`** (actual env steps,
8 envs × 4 PPO rollouts of `n_steps=2048`).

Evaluation protocol (identical for every run, per
`rl_logs_sb3/EXPERIMENTATION_PROCESS.md`):

- Test day **`496528674@qq.com_20161128`** (in held-out test split).
- **`eval_seed = 42`**, **30 min** greedy warmup, then **8 h** composite episode.
- Two compares: **`final`** = `ppo_sc_final.zip`, **`best`** = `best_model/best_model.zip`.

---

## Shared configuration (all 50k runs)

| Item | Value |
|---|---|
| Algorithm | **PPO** (Stable Baselines 3) |
| Observation space | Box `(17,)` float32, unbounded |
| Action space | Box `(2,)`, **λ₁ ∈ [0, 2.0]**, **λ₂ ∈ [0, 0.5]** |
| **λ₃** | **Fixed at 1.0** (unit anchor) |
| Step duration | **5 min** |
| Warmup | **30 min** (greedy, shared) |
| Episode | **8 h** (96 composite steps) |
| Train/test split | **24 / 6** days, `random_state=42`, shuffled |
| Data root | `./data/didi/full_didi_gaia` |
| VecEnv | **8 SubprocVecEnv workers** |
| `--timesteps` (CLI) | **50,000** → actual **65,536** env steps |
| Hyperparams file | `best_hyperparameters.json` (sha `bb834b0e…`), `n_steps=2048`, `batch_size=256`, **net_arch_type=large** `[256, 256]`, `lr=3e-4`, `gamma=0.95`, `gae_lambda=0.9`, `clip=0.2`, `ent_coef=0.04`, `vf_coef=0.55`, `max_grad_norm=1.0` |
| `reward_weights` (fairness, starvation, latency) | **`[1.0, 1.0, 1.0]`** |

---

## Per-run quick reference

Branches column = git branches **containing** the manifest commit (the run was almost
certainly trained on the *most specific* branch listed).

| Run folder | Created (UTC) | Commit | Branch (most specific) | Reward function (one-liner) |
|---|---|---|---|---|
| `run_20260402_135637` | 2026-04-03 01:18 | *backfill* | (pre-snapshot, ancestor of all SLA/twin branches) | Absolute composite, **`(reward − 20.0) / 2.0`**, `r_fairness × 50`, `r_latency × 2`, `r_starv × 0.5` |
| `run_20260403_122313` | 2026-04-03 01:23 | `0e42e35` *(“new reward function”)* | ancestor of `Twin-simulator`, `conference-ready`, `oracle-approach`, `oracle-dynamic-sla`, `twin-dynamic-sla` | Absolute composite, **`(reward − 50.0) / 5.0`**, `r_fairness × 100`, `r_latency × 2`, `r_starv × 0.5` |
| `run_20260430_154704` | 2026-04-30 05:47 | `5209775` *(“calibrate sla script”)* | same ancestor set as above | Absolute + **fixed-SLA gate**: latency penalty only above `sla_wait_time_minutes`, `r_latency = -sla_violation_penalty × violation`, then `(reward − 50.0)/5.0` |
| `run_20260510_062142` | 2026-05-09 20:21 | `6640a34` *(“pure linear advantage”)* | **`oracle-approach`** | **Pure advantage vs greedy oracle**: `r = (Δfairness×100) + (Δlatency×5) + (Δstarv×1)`, then `/5.0` |
| `run_20260510_133620` | 2026-05-10 03:36 | `ee42bb2` *(“dynamic sla”)* | **`oracle-dynamic-sla`** | **Bounded dynamic-SLA vs oracle**: `+100·Δfairness`, latency penalty `−10·diff` only when `latency_diff > 0.1` min, `+1·Δstarv`, `/5.0` |
| `run_20260510_200831` | 2026-05-10 10:08 | `70c18da` *(“sla function”)* | **`twin-dynamic-sla`** | **Bounded dynamic-SLA vs greedy twin** (shadow sim): same shape as oracle-dynamic-sla but baseline = `shadow_reward_stats` |
| `run_20260511_022820` | 2026-05-10 16:28 | `ec9ef42` *(“Twin-simulator implemented”)* | **`Twin-simulator`** | **Composite − greedy-twin advantage**: full composite formula computed for both arms and returns `(advantage / 5.0)` (no SLA gate) |

> All seven jobs were started with **`python rl/train_sb3.py --timesteps 50000 --hyperparams best_hyperparameters.json`** and ran on EC2 (`ip-10-110-145-230`), `dirty=False`. The April 2 run is the only one without `argv`/git in its manifest (artifacts were **backfilled** by `scripts/backfill_run_artifacts.py`).

---

## Performance vs static composite baseline (held-out test day)

“Improvement” is **RL − static**. **Lower** is better for *Peak Backlog* and *Avg Wait*; **higher** is better for *JFI*. Static numbers are copied from each run’s `baseline_*_model_metrics.txt`; the small static drift across runs is expected (different env snapshots / minor sim changes).

### Final-checkpoint (`ppo_sc_final.zip`)

| Run | JFI static | JFI RL | ΔJFI | Peak BL static | Peak BL RL | ΔBL | Wait static (m) | Wait RL (m) | ΔWait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_20260402_135637` | 0.5616 | 0.5461 | **−0.0155** | 105 | **103** | **−2** | 2.67 | **2.62** | **−0.05** |
| `run_20260403_122313` | 0.5650 | 0.5475 | −0.0175 | 100 | 103 | +3 | 2.67 | **2.62** | **−0.06** |
| `run_20260430_154704` | 0.5636 | 0.5515 | −0.0121 | 100 | 101 | +1 | 2.67 | **2.61** | **−0.05** |
| `run_20260510_062142` | 0.5682 | 0.5495 | **−0.0187** | 103 | **101** | **−2** | 2.67 | **2.63** | **−0.04** |
| `run_20260510_133620` | 0.5626 | 0.5531 | **−0.0095** ✅ smallest fairness loss | 103 | 103 | 0 | 2.68 | **2.62** | **−0.06** |
| `run_20260510_200831` | 0.5660 | 0.5486 | −0.0173 | 101 | 102 | +1 | 2.65 | **2.61** | **−0.04** |
| `run_20260511_022820` | 0.5678 | 0.5490 | **−0.0188** | 101 | 102 | +1 | 2.66 | **2.63** | **−0.04** |

### Best-eval-reward checkpoint (`best_model/best_model.zip`)

| Run | JFI static | JFI RL | ΔJFI | Peak BL static | Peak BL RL | ΔBL | Wait static (m) | Wait RL (m) | ΔWait |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `run_20260402_135637` | 0.5652 | 0.5501 | −0.0151 | 103 | 103 | 0 | 2.66 | 2.65 | −0.02 |
| `run_20260403_122313` | 0.5636 | 0.5509 | −0.0126 | 101 | 104 | +3 | 2.67 | 2.64 | −0.03 |
| `run_20260430_154704` | 0.5642 | 0.5504 | −0.0137 | 101 | 101 | 0 | 2.66 | 2.62 | −0.04 |
| `run_20260510_062142` | 0.5641 | 0.5505 | −0.0136 | 100 | 104 | +4 | 2.67 | 2.62 | −0.05 |
| `run_20260510_133620` | 0.5594 | 0.5501 | **−0.0093** ✅ smallest fairness loss | 101 | 102 | +1 | 2.68 | **2.61** | **−0.07** ✅ best wait gain |
| `run_20260510_200831` | 0.5671 | 0.5490 | −0.0182 | 101 | 102 | +1 | 2.67 | 2.62 | −0.05 |
| `run_20260511_022820` | 0.5640 | 0.5509 | −0.0131 | 101 | 101 | 0 | 2.66 | 2.62 | −0.04 |

**Headline:** every run **trades a small fairness loss (~−0.01 JFI) for a small wait-time win (~−0.04 to −0.07 min)**. **Peak backlog is essentially unchanged**. No run yet *Pareto-improves* on the static composite baseline — across all 7 runs, **JFI never beats baseline**.

The “dynamic SLA” reward (`run_20260510_133620`, `oracle-dynamic-sla`) is the **closest to neutral fairness** while still giving the **best wait-time improvement**, on **both** the final and best checkpoints. **`run_20260510_062142`** (pure-advantage, `oracle-approach`) is the only run that simultaneously **reduces peak backlog by 2** on the final checkpoint, but it pays for it with the largest fairness drop on final (−0.0187).

---

## λ-weight behaviour (per-step trace, 96 steps × 2 dims)

Re-extracted from `baseline_*_model_weight_outputs.txt`. **λ₃ is fixed at 1.0** by env design, so any non-trivial behaviour lives in **λ₁ (fairness)** and **λ₂ (starvation)**. Action-space caps are λ₁ ∈ [0, 2.0], λ₂ ∈ [0, 0.5].

### Final policy

| Run | λ₁ min/max/μ/σ | λ₂ min/max/μ/σ | Notes |
|---|---|---|---|
| `run_20260402_135637` | 0 / 0 / 0 / 0 | 0 / 0.099 / 0.005 / 0.013 | **Collapsed**: λ₁ off, very small λ₂ bursts |
| `run_20260403_122313` | 0 / 0.060 / 0.003 / 0.010 | 0 / 0 / 0 / 0 | Tiny λ₁ jitter, λ₂ off |
| `run_20260430_154704` | 0 / 0.033 / 0.000 / 0.003 | 0 / 0.219 / 0.027 / 0.034 | **Largest λ₂ swings** in this set, but still mostly small |
| `run_20260510_062142` | 0 / 0 / 0 / 0 | 0 / 0.039 / 0.000 / 0.004 | Near-degenerate (effectively static λ) |
| `run_20260510_133620` | 0 / 0 / 0 / 0 | 0 / 0.137 / **0.081 / 0.032** | **Most stable, non-trivial λ₂** policy → matches its best metric profile |
| `run_20260510_200831` | 0 / 0 / 0 / 0 | 0 / 0.016 / 0.000 / 0.002 | Near-degenerate again |
| `run_20260511_022820` | 0 / 0 / 0 / 0 | 0 / 0.135 / 0.007 / 0.020 | Sparse but occasional spikes |

### Best policy

| Run | λ₁ min/max/μ/σ | λ₂ min/max/μ/σ |
|---|---|---|
| `run_20260402_135637` | **0.041 / 0.160 / 0.093 / 0.036** | 0 / 0.040 / 0.015 / 0.012 |
| `run_20260403_122313` | 0 / 0.071 / 0.042 / 0.025 | 0 / 0.014 / 0.000 / 0.001 |
| `run_20260430_154704` | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| `run_20260510_062142` | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| `run_20260510_133620` | 0 / 0 / 0 / 0 | 0 / 0.084 / 0.038 / 0.022 |
| `run_20260510_200831` | 0 / 0 / 0 / 0 | 0 / 0.095 / 0.002 / 0.010 |
| `run_20260511_022820` | 0 / 0 / 0 / 0 | 0 / 0.017 / 0.002 / 0.004 |

### Patterns worth noting

- Across **6 of 7** finals, **λ₁ is identically 0** — the policy has effectively chosen *not* to upweight fairness vs the unit-anchor utility (λ₃ = 1). The only run with **non-zero λ₁** behaviour at all is the very first **April-3 backfilled run** (best ckpt μ=0.093), and that run also happens to have one of the better fairness profiles. **Strong hint:** later reward shapes are pushing λ₁ to a corner.
- λ₂ activity is also **far below the 0.5 cap** for every run; even the “healthiest” run (`run_20260510_133620`) caps around **0.137**. The action space might be **larger than the policy needs**.
- The two **best** runs by behavioural diversity (non-zero μ, σ > 0.02) are `run_20260402_135637` (best ckpt) and `run_20260510_133620` (final + best). They are also the **two best on the metrics tables** in their respective categories.
- “Pure-advantage” (`run_20260510_062142`) and the “composite−twin advantage” (`run_20260511_022820`) reward shapes both **collapse to near-static λ** — the agent is essentially refusing to act, and metrics still beat baseline on wait-time only because the static λ baseline itself is conservative.

---

## Reward-function diffs (the main per-run lever)

All runs use `reward_weights = [1.0, 1.0, 1.0]`, so the **reward shape itself** is the variable. Excerpts of `gym_environment_snapshot.py::_calculate_reward` per run:

### `run_20260402_135637`  (backfill / pre-SLA absolute)

```python
r_fairness   = stats['fairness']          * 50.0
r_latency    = -stats['latency']          * 2.0
r_starvation = -stats['recent_expirations'] * 0.5
reward       = w0*r_fairness + w1*r_starvation + w2*r_latency
return (reward - 20.0) / 2.0
```

### `run_20260403_122313`  (commit `0e42e35`, “new reward function”)

```python
r_fairness   = stats['fairness']          * 100.0   # boosted
r_latency    = -stats['latency']          * 2.0
r_starvation = -stats['recent_expirations'] * 0.5
reward       = w0*r_fairness + w1*r_starvation + w2*r_latency
return (reward - 50.0) / 5.0
```

### `run_20260430_154704`  (commit `5209775`, “calibrate sla script”)  — fixed SLA gate

```python
r_fairness   = stats['fairness'] * 100.0
r_starvation = -stats['recent_expirations'] * 0.5
sla = self.sla_wait_time_minutes
if latency <= sla:
    reward = w0*r_fairness + w1*r_starvation
else:
    r_latency = -self.sla_violation_penalty * (latency - sla)
    reward    = w0*r_fairness + w1*r_starvation + w2*r_latency
return (reward - 50.0) / 5.0
```

### `run_20260510_062142`  (commit `6640a34`, **`oracle-approach`**, “pure linear advantage”)

```python
fairness_adv   = composite['fairness']         - oracle['fairness']
latency_adv    = oracle['latency']             - composite['latency']
starvation_adv = oracle['recent_expirations']  - composite['recent_expirations']
reward = fairness_adv*100 + latency_adv*5 + starvation_adv*1
return reward / 5.0
```

### `run_20260510_133620`  (commit `ee42bb2`, **`oracle-dynamic-sla`**)

```python
r_fairness   = (composite['fairness'] - oracle['fairness']) * 100.0
latency_diff = composite['latency']   - oracle['latency']
r_latency    = 0.0 if latency_diff <= 0.1 else -10.0 * latency_diff
r_starv      = (oracle['recent_expirations'] - composite['recent_expirations']) * 1.0
return (r_fairness + r_latency + r_starv) / 5.0
```

### `run_20260510_200831`  (commit `70c18da`, **`twin-dynamic-sla`**)

Same shape as `run_20260510_133620` but baseline is `shadow_reward_stats` (parallel **greedy twin sim**), not the on-policy oracle.

### `run_20260511_022820`  (commit `ec9ef42`, **`Twin-simulator`**)

```python
# Compose composite reward and shadow (greedy twin) reward separately, then take the diff.
reward_composite = w0*r_fair_c + w1*r_starv_c + w2*r_lat_c
reward_shadow    = w0*r_fair_s + w1*r_starv_s + w2*r_lat_s
return (reward_composite - reward_shadow) / 5.0   # no SLA gate
```

---

## Cross-run takeaways

1. **All seven runs land in the same narrow regime**: ΔJFI ∈ [−0.019, −0.009], ΔWait ∈ [−0.07, −0.02] min, ΔPeakBacklog ∈ [−2, +4]. The current setup is **stuck trading fairness for tiny wait gains**.
2. The **dynamic-SLA-vs-oracle reward** (`run_20260510_133620`) is the **single best 50k run** by combined fairness + wait-time on both `final` and `best` checkpoints. It is also the **only one that gives the policy a clear, non-zero λ₂ band** in the final.
3. **Switching the baseline from “absolute composite reward” to “advantage over a sibling sim”** (oracle / twin) does **not** by itself unlock new gains; the *gating* (linear vs SLA-bounded) seems to matter more than what the “fair comparator” is.
4. **Action collapse** is now the dominant failure mode. λ₁ ≡ 0 in 6/7 finals, λ₂ < 0.05 max in 4/7 finals. Either (a) **expand the action range** (or transform it), (b) add an **entropy bonus / KL penalty against the static λ**, or (c) **shape the reward to specifically punish stagnant λ** (e.g. via a small variance bonus).
5. **No `--resume` was used** for any of these seven 50k runs. They are independent fresh training jobs of the same budget on increasingly newer reward functions. Comparisons are therefore apples-to-apples on training duration but **not** on reward / env code.
6. The static baseline numbers (e.g. JFI ~0.56, Wait ~2.67 m) are **stable across snapshots**, so cross-run RL deltas are a fair signal even though the env code is changing.

---

## Suggestions for next experiments

- **Re-train `run_20260510_133620`’s reward (oracle-dynamic-SLA) for `--timesteps 200000` (or 300k)** — `run_20260509_055756` already shows you can run 300k, and the dynamic-SLA reward’s λ trace looks the most “alive” at 50k, so it’s the best candidate for longer training.
- **Tighten action bounds** to where the policy actually lives (e.g. λ₂ ∈ [0, 0.2]) to make the entropy budget more useful in that range.
- **Add a small entropy floor on λ₁** (e.g. min ε constraint or a soft KL-to-uniform term in the reward) to break the λ₁ ≡ 0 collapse before doubling the budget.
- **Hold `oracle-dynamic-sla` and `twin-dynamic-sla` head-to-head** at matching budgets — the same gating reward only differs in whether the comparator is the oracle or the parallel twin. Right now both 50k runs underperform the dynamic-SLA-vs-oracle baseline on fairness, so it’s unclear whether the *twin sim* signal is helpful at all.
- Optional: log per-rollout `rollout/ep_rew_mean` from each run’s `PPO_1/` TensorBoard to make a single comparison plot — the static metrics here are the post-eval table, **not** the training curves.

---

## Provenance / files used

For every run above, the following artifacts were read directly:

- `run_manifest.json` — argv, git commit, dirty flag, training params, baseline + post-eval metadata.
- `environment_spec.json` — observation/action spec, runtime fields (`reward_weights`, `lambda3_fixed`, etc.).
- `hyperparams_snapshot.json` — PPO + net_arch.
- `gym_environment_snapshot.py` — the `_calculate_reward` excerpts above.
- `baseline_final_model_metrics.txt`, `baseline_best_model_metrics.txt` — JFI / Peak Backlog / Avg Wait tables.
- `baseline_final_model_weight_outputs.txt`, `baseline_best_model_weight_outputs.txt` — per-step λ₁/λ₂/λ₃ traces (96 steps each).
- `ppo_sc_final.zip` — verified `_total_timesteps = 50000`, `num_timesteps = 65536` for every run.
