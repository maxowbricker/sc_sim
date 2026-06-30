# §5.3 — Computational Efficiency & Scalability

**Claim:** k-NLF and Composite are O(k log k) per event — their wall-clock runtime stays
flat as fleet size scales, while LAF (O(W)) and FATP-ANN (full scan) grow super-linearly.

**Results output:** `results/s53_scalability/`

---

## Run commands

```bash
# Experiment A: Vary fleet size |W|, fix task count
caffeinate python scripts/experiments/s53_scalability/run_scalability_fleet.py \
    > results/s53_scalability/log_fleet.log 2>&1

# Experiment B: Vary task volume |T|, fix fleet size
caffeinate python scripts/experiments/s53_scalability/run_scalability_tasks.py \
    > results/s53_scalability/log_tasks.log 2>&1
```

Run both concurrently on the cluster (they are independent):

```bash
python scripts/experiments/s53_scalability/run_scalability_fleet.py > results/s53_scalability/log_fleet.log 2>&1 &
python scripts/experiments/s53_scalability/run_scalability_tasks.py > results/s53_scalability/log_tasks.log 2>&1 &
wait
echo "Scalability experiments complete"
```

---

## Output files
| File | Contents |
|------|----------|
| `results/s53_scalability/scalability_fleet.csv` | Fleet size × strategy × wall-clock time |
| `results/s53_scalability/scalability_tasks.csv` | Task volume × strategy × wall-clock time |

---

## Expected runtime growth patterns

| Strategy | Complexity | Expected behaviour as |W| grows |
|----------|-----------|-------------------------------|
| k-NLF | O(k log k) per event | Flat — independent of \|W\| |
| Composite | O(k log k) per event | Flat — independent of \|W\| |
| Greedy | O(W) per event | Linear growth |
| LAF (LTF) | O(W) per event | Linear growth |
| FATP-ANN | O(k) with full ANN build | Near-linear to super-linear |
| Discrete Review LP | O(W×T) per review epoch | Super-linear / exponential |

---

## Sweep parameters

**Experiment A — Fleet size sweep**
- Worker counts: 1,000 / 5,000 / 10,000 / 20,000 / 40,000
- Tasks: fixed at stratified sample matching the 1:6 ratio of the 40k-worker config
- Strategies: k-NLF, Composite, Greedy, LAF, FATP-ANN (Discrete Review LP optional)

**Experiment B — Task volume sweep**
- Task counts: 10,000 / 40,000 / 80,000 / 140,000 / 200,000
- Workers: fixed at 10,000
- Same strategy set

---

## Estimated wall-clock time (laptop baseline)
- Experiment A: ~3.5–4 h sequential
- Experiment B: ~1.5–2 h sequential
- On C33 cluster (8 CPUs): both run concurrently → ~4 h total
