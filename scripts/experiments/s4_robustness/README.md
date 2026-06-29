# Section 4 — Robustness to Spatiotemporal Density

Tests whether the ranking of strategies (and Composite's gains) holds across varying
worker:task supply/demand ratios. Also includes wall-time analysis for the efficiency
discussion.

## Scripts in this directory

| Script | What it does |
|--------|-------------|
| `run_didi_ratio_sweep.py` | Didi at 1:4 / 1:5 / 1:7 ratios × {Greedy, LAF, ONRTA-OP, Composite} |

Gowalla ratio sweep uses **`scripts/run_gowalla_comparison.py`** with its built-in
`--ratio` flag (see run commands below).

## Run commands

```bash
# Didi ratio sweep — ~20 min
python scripts/experiments/s4_robustness/run_didi_ratio_sweep.py

# Gowalla ratio sweep: 1:7 (default), 1:5, 1:4
python scripts/run_gowalla_comparison.py --ratio 0.143 \
    --output results/s4_robustness/gowalla_ratio_0143.csv
python scripts/run_gowalla_comparison.py --ratio 0.2 \
    --output results/s4_robustness/gowalla_ratio_020.csv
python scripts/run_gowalla_comparison.py --ratio 0.25 \
    --output results/s4_robustness/gowalla_ratio_025.csv
```

## Output files
| File | Contents |
|------|----------|
| `results/s4_robustness/didi_ratio_sweep.csv` | 3 ratios × 4 strategies — TAR, JFI, wait, elapsed_s |
| `results/s4_robustness/gowalla_ratio_*.csv` | Gowalla at each density — full strategy table |

## Computational efficiency analysis
The `elapsed_s` column in `didi_ratio_sweep.csv` and in the Section 1 full-strategy
runs feeds directly into the efficiency scatter plot (wall time vs. TAR or JFI).

Expected complexity:
| Strategy | Complexity per event | Notes |
|---|---|---|
| Greedy | O(k) | k-NN lookup only |
| EWMA-Only | O(k) | same scan as Greedy |
| LAF | O(W) | scans all workers |
| Composite | O(k) | k-NN + O(k) scoring |
| ONRTA-OP | O(W·T) | offline LP-style |
| BiRanking | O(W) | ranking pass |
| FATP-ANN | O(k) | approximate ANN |

## Paper narrative
> "Across 1:4 to 1:7 worker:task ratios, Composite maintains its JFI advantage over
> Greedy (Δ JFI ≥ X) with no significant increase in average wait time. Composite
> achieves this with O(k) per-event complexity — identical to Greedy — making it
> well-suited for high-throughput deployment."
