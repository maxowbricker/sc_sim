# Git branches — onboarding guide

Use this map when checking out code for training, baselines, or reproducing a past run. Run folders under `rl_logs_sb3/run_*` store **`gym_environment_snapshot.py`** and **`run_manifest.json`** (`git.commit` SHA) — those are the ground truth for what code produced a result. **Renaming branches does not change commit hashes.**

> **Note:** The primary RL branch was formerly named **`conference-ready`**. Older run manifests and experiment docs may still say `conference-ready`; use the recorded **commit SHA** or the run-folder snapshots to reproduce.

---

## Branches (current)

| Branch | Purpose |
|--------|---------|
| **`main`** | **Start here.** Primary RL codebase (standard reward formulations, ΔJFI, etc.). Formerly `conference-ready`. |
| **`event-driven`** | Pre-RL honours archive (event-driven simulator, deferral tracker, early strategies). GitHub default for historical repo layout. |
| **`oracle-approach`** | Oracle reward line: greedy one-step counterfactual per env step; best-performing reward family in recent experiments. |

Training entrypoint (from repo root):

```bash
git checkout main
python rl/train_sb3.py --timesteps 50000 --hyperparams best_hyperparameters.json
```

See `rl_logs_sb3/EXPERIMENTATION_PROCESS.md` for run artifacts and baseline eval protocol.

---

## Reproducing a past run

1. Open `rl_logs_sb3/run_YYYYMMDD_HHMMSS/run_manifest.json` → note **`git.commit`**.
2. Either:
   - `git checkout <commit-sha>` in the repo, or
   - Use **`gym_environment_snapshot.py`** / **`train_sb3_snapshot.py`** in that run folder (self-contained copies).

Branch names in old docs are informational only; **commit SHA + snapshots** are authoritative.

---

## Archive tags (deleted experiment branches)

These branches were removed to simplify onboarding; tags preserve the tips:

| Tag | Former branch |
|-----|----------------|
| `archive/main-pre-rl` | Old deleted `main` (tick-loop sim; superseded by `event-driven`) |
| `archive/oracle-dynamic-sla` | Oracle + capped dynamic SLA reward |
| `archive/oracle-static-composite` | Oracle = static composite [1.0, 0.2, 1.0] |
| `archive/twin-simulator` | Twin greedy shadow simulator |
| `archive/twin-dynamic-sla` | Twin + dynamic SLA |
| `archive/multi-channel-feature-maps` | Old observation-space experiment |

```bash
git checkout archive/oracle-static-composite   # detached HEAD at old tip
```
