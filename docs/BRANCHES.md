# Git branches — onboarding guide

Use this map when checking out code for training, baselines, or reproducing a past run. Run folders under `rl_logs_sb3/run_*` store `gym_environment_snapshot.py` and `run_manifest.json` (git commit) — those are the ground truth for what code produced a result.

---

## Start here

| Branch | Purpose |
|--------|---------|
| **`conference-ready`** | **Default RL codebase.** Standard reward formulations (absolute / ΔJFI, no per-step oracle counterfactual). Most training runs and docs assume this branch. **New collaborators should clone and checkout this branch first.** |

Training entrypoint (from repo root):

```bash
python rl/train_sb3.py --timesteps 50000 --hyperparams best_hyperparameters.json
```

See `rl_logs_sb3/EXPERIMENTATION_PROCESS.md` (or copy in `docs/`) for run artifacts and baseline eval protocol.

---

## Archive

| Branch / tag | Purpose |
|--------------|---------|
| **`main`** | Pre-RL simulation baseline (~11 months stale vs `conference-ready`). Kept for historical comparison; **do not use for new RL work.** |
| Tag **`archive/main-pre-rl`** | Bookmark on `main` tip before RL fork. |
| Tag **`archive/multi-channel-feature-maps`** | Bookmark on old observation-space experiment (`Multi-Channel-Feature-Maps`). |

---

## Oracle reward experiments (most promising line)

These branches share the same simulator and training pipeline as `conference-ready`, but **`rl/gym_environment.py`** runs a **counterfactual reference step** each env step and computes reward as **RL vs reference** advantage.

| Branch | Oracle reference | Reward idea |
|--------|------------------|-------------|
| **`oracle-approach`** | **Greedy** one-step roll-forward from same state | Linear / asymmetric advantage vs greedy (best overall family so far; see `docs/experiments/ORACLE_50K_COMPARISON.md`) |
| **`oracle-dynamic-sla`** | Greedy oracle | Dynamic SLA: latency penalty only when RL is >0.1 min slower than oracle; capped penalty floor |
| **`oracle-static-composite`** | **Static composite** fixed λ₁=1.0, λ₂=0.2, λ₃=1.0 | Symmetric advantage vs the static heuristic the paper aims to beat |

`oracle-approach` has **diverged** from `conference-ready` (not simply “a few commits behind”) — merge carefully or cherry-pick reward changes.

---

## Twin-simulator experiments

Parallel **shadow simulator** (greedy twin) instead of single-step greedy oracle:

| Branch | Notes |
|--------|--------|
| **`Twin-simulator`** | Twin runs greedy in parallel; composite-minus-twin advantage reward |
| **`twin-dynamic-sla`** | Twin + dynamic-SLA-style latency gating |

Use when ablating oracle vs twin architecture; tied to specific `run_*` folders from May 2026 sprint.

---

## Removed / merged (safe to ignore)

These local branches were deleted during repo cleanup; their work lives on **`conference-ready`** or tags:

| Former branch | Fate |
|---------------|------|
| `conference-readyu` | Typo duplicate of `potential-shaping` — deleted |
| `potential-shaping` | Merged into `conference-ready` — deleted (commit `9d2b0ab` on conference-ready history) |
| `RL` | Early RL experiments, merged — deleted |

---

## Remote default branch

GitHub **`origin/HEAD`** may still point at legacy branches (`event-driven` or `main`). For development, **checkout `conference-ready` explicitly** until the repo default is updated to `conference-ready` in GitHub Settings.

---

## Quick reference: which branch for what?

| Goal | Branch |
|------|--------|
| Reproduce standard PPO / ΔJFI runs | `conference-ready` |
| Reproduce oracle 50k suite | `oracle-approach`, `oracle-dynamic-sla`, `oracle-static-composite` |
| Old simulation-only code | `main` (or tag `archive/main-pre-rl`) |
| Understand a specific run | Read that run’s `run_manifest.json` + `gym_environment_snapshot.py`, then checkout matching commit or branch |
