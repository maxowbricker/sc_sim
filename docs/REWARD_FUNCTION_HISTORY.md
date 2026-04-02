# Reward function history (`rl/gym_environment.py`)

This log was generated from **git history** of `rl/gym_environment.py`.  
**Scope:** commits from **2026-03-21** through **2026-03-31** (~10 days ending **2026-03-31**), plus **optional context** for the last change before that window.

---

## How this was produced

```bash
git log --since="2026-03-21" --reverse --format="%h %ad %s" --date=short -- rl/gym_environment.py
git show <commit>:rl/gym_environment.py  # extract `_calculate_reward`
```

**Current repo state** (`HEAD` at time of writing): matches **2026-03-31** commit `a1803a7` for `_calculate_reward` (no uncommitted changes).

---

## Timeline (10-day window: 2026-03-21 to 2026-03-31)

| Date | Commit | Short message | Reward changed? |
|------|--------|----------------|-----------------|
| 2026-03-28 | `fe8c037` | removal of hardcoded strategy param and last action from config | **No** (same as below) |
| 2026-03-29 | `e050ed7` | compare baseline script + episode duration changes | **No** |
| 2026-03-29 | `d9155e4` | metric scaling → `OBSERVATION_STATIC_SCALING` | **No** |
| 2026-03-29 | `6edc357` | removal of assignment delay from observation space | **No** |
| 2026-03-30 | `fbdf59f` | updates to observation scaling and reward function | **Yes** → momentum |
| 2026-03-31 | `a1803a7` | changes in reward function | **Yes** → absolute |

**Note:** There were **no** commits to `rl/gym_environment.py` between **2026-03-21** and **2026-03-27**. Through that period the reward stayed on the **exponential vs greedy warmup JFI** formulation (see Variant A).

---

## Variant A — Exponential fairness vs greedy warmup JFI (~2026-03-20 → 2026-03-29)

**Commits:** `7dbc228` … `6edc357` (unchanged from `fe8c037^` through `6edc357` for reward).

**Idea:** Fairness = improvement of **current JFI** over **`self.greedy_baseline_jfi`** (snapshot at end of greedy warmup). Positive gap: linear `× 20`; negative gap: exponential penalty.

```python
def _calculate_reward(self):
    stats = self.simulator.metrics.get_reward_stats(self.simulator.current_time)

    current_jfi = stats['fairness']
    jfi_improvement = current_jfi - self.greedy_baseline_jfi

    if jfi_improvement >= 0:
        r_fairness = jfi_improvement * 20.0
    else:
        r_fairness = -10.0 * (np.exp(abs(jfi_improvement) * 5.0) - 1.0)

    r_latency = -stats['latency'] / 5.0
    r_starvation = -min(3.0, stats['recent_expirations'] / 20)

    reward = (self.reward_weights[0] * r_fairness) + \
             (self.reward_weights[1] * r_starvation) + \
             (self.reward_weights[2] * r_latency)

    return float(reward)
```

---

## Variant B — Momentum ΔJFI (2026-03-30, `fbdf59f`)

**Idea:** Fairness = **change in JFI since the previous step** (`reward_prev_jfi` updated each step; initialized at handover in `reset()`). Scales `delta_jfi` by **33.3** (~0.03 bump → ~+1.0 raw). Latency `× 0.5`; starvation capped with `min(3, …)`.

```python
def _calculate_reward(self):
    stats = self.simulator.metrics.get_reward_stats(self.simulator.current_time)

    current_jfi = stats['fairness']
    delta_jfi = current_jfi - self.reward_prev_jfi
    self.reward_prev_jfi = current_jfi
    r_fairness = delta_jfi * 33.3

    r_latency = -stats['latency'] * 0.5
    r_starvation = -min(3.0, stats['recent_expirations'] * 0.5)

    reward = (self.reward_weights[0] * r_fairness) + \
             (self.reward_weights[1] * r_starvation) + \
             (self.reward_weights[2] * r_latency)

    return float(reward)
```

---

## Variant C — Absolute JFI / wait / expirations (2026-03-31, `a1803a7`, current `HEAD`)

**Idea:** Fairness = **raw JFI** (not vs baseline, not Δ). **No** use of `greedy_baseline_jfi` in the reward (still computed in `reset()` for logging/other use). Latency `× 2`; starvation linear, **uncapped** (no `min` on starvation term).

```python
def _calculate_reward(self):
    stats = self.simulator.metrics.get_reward_stats(self.simulator.current_time)

    r_fairness = stats['fairness'] * 50.0
    r_latency = -stats['latency'] * 2.0
    r_starvation = -stats['recent_expirations'] * 0.5

    reward = (self.reward_weights[0] * r_fairness) + \
             (self.reward_weights[1] * r_starvation) + \
             (self.reward_weights[2] * r_latency)

    return float(reward)
```

---

## Appendix — Earlier experiment (before the 10-day window)

**Commit `9607285` (2026-03-19)** — “updates to the reward function” — used a **fixed 0.5 center**, a **cliff** below JFI 0.75, and different starvation scaling. This was **replaced** by the exponential **vs greedy warmup** design by **2026-03-20** (`7dbc228`).

```python
def _calculate_reward(self):
    stats = self.simulator.metrics.get_reward_stats(self.simulator.current_time)

    r_fairness = (stats['fairness'] - 0.5) * 10.0
    if stats['fairness'] < 0.75:
        r_fairness -= 20.0  # The Cliff

    r_latency = -stats['latency'] / 5.0
    r_starvation = -(stats['recent_expirations'] / 10.0)

    reward = (self.reward_weights[0] * r_fairness) + \
             (self.reward_weights[1] * r_starvation) + \
             (self.reward_weights[2] * r_latency)

    return reward
```

---

## Summary

| Variant | Period (approx) | Fairness signal |
|--------|------------------|-----------------|
| **D** (appendix) | ≤ 2026-03-19 | Absolute vs 0.5 + cliff at 0.75 |
| **A** | 2026-03-20 → 2026-03-29 | vs **greedy warmup JFI**, piecewise linear / exponential |
| **B** | 2026-03-30 | **ΔJFI** vs previous step |
| **C** | 2026-03-31+ | **Absolute JFI** × 50 |

---

*Regenerate or extend this file after future reward edits with `git log -p -- rl/gym_environment.py`.*
