# Section 1 — Overall Strategy Performance

Scripts in this section live in the **root `scripts/` directory** (they predate section
organisation and are kept there to avoid breaking existing invocations).

## Run commands

```bash
# Didi 20161109 — full 10-strategy comparison
caffeinate python scripts/run_strategy_comparison.py \
    --day 20161109 \
    --output results/s1_overall_performance/didi_20161109.csv

# Gowalla Austin compressed — full strategy comparison
caffeinate python scripts/run_gowalla_comparison.py \
    --compression compressed \
    --output results/s1_overall_performance/gowalla_austin_compressed.csv
```

## Output files
| File | Contents |
|------|----------|
| `results/s1_overall_performance/didi_20161109.csv` | One row per strategy — TAR, JFI, wait, utilisation |
| `results/s1_overall_performance/gowalla_austin_compressed.csv` | Same columns, Gowalla dataset |

## Key results notes
- **Composite (static)**: weights fw=1.0 sw=0.2 uw=1.0 k=15 γ=0.1 st=0.05
  - Same weights for Didi *and* Gowalla — generalisation claim
- Cost-Balancing, TSGF Sampling, MMD-Batch **excluded** (time out on 20161109)
- LAF was re-added after being missing from earlier runs
- EWMA-Only was broken (float timestamp bug) — fixed Jun 29 2026, needs rerun

## Gowalla dataset setup
Full setup documentation (compression rationale, exact config, first run results):
→ [`docs/datasets/gowalla_austin_setup.md`](../../../docs/datasets/gowalla_austin_setup.md)
