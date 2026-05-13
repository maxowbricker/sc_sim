# Assessment: Section 5.1 (Experimental Setup)

**Overall Quality:** ✅ **Strong foundation. Well-structured, detailed, and clear.**

This section is significantly better than many conference papers—you've included specific numbers, design rationales, and careful attention to temporal leakage. Below are gaps, strengths, and a clear recommendation on the Evaluation Metrics subsection.

---

## 1. What's Working Well

### ✅ Dataset and Regional Realization
- Exact numbers (7.06M trips, 1.09B GPS pings, 30.67°N, 104.07°E)
- Clear justification for flat-earth approximation (< 0.15% error)
- Proportional stratified temporal sampling is **excellent** — it preserves demand peaks/lulls
- Citation to Tong et al. (2019) survey establishes context

### ✅ Training and Evaluation Protocol
- Day-level temporal split (80/20) prevents leakage ✅
- 30-minute greedy warmup addresses cold-start problem ✅
- 96 decision steps per episode (5-min granularity over 8 hours) is very clear
- Fixed random seed for reproducibility (good implicit assumption)

### ✅ Simulation Physics
- Discrete Event Simulation architecture justified
- Manhattan distance + 30 km/h constant speed is standard and reasonable
- Dual-condition feasibility check (pickup before expiry + completion before shift deadline) is well-explained
- Unix timestamps for temporal fidelity is strong

### ✅ Baselines
- Good mix: greedy (lower bound), LAF (fairness), FATP-ANN (related work), Fixed-Composite (static version), MMD-BB (bottleneck fairness)
- Each baseline has clear motivation
- Citations to CR-08 and CR-11

---

## 2. Important Gaps / Missing Details

### 🔴 **CRITICAL:** DRL Agent Hyperparameters
**Missing:** Your paper defines baselines precisely, but the DRL agent setup is vague.

**Should add:**
- Network architecture (e.g., "Two fully-connected layers with 256 units each, ReLU activations")
- PPO-specific hyperparameters: learning rate, batch size, clip range, GAE λ, entropy coefficient
- Training hardware: GPU? CPU? How many parallel environments?
- How many training episodes before evaluation? (You mention 24 training days, but is that 1 episode per day repeated N times, or what?)
- Random seed for reproducibility

**Where to add:** New bullet under "Training and Evaluation Protocol" or separate "Agent Configuration" subsubsection.

**Suggested text:**
> "The DRL Governor uses Proximal Policy Optimization (PPO) with a two-layer fully-connected policy network (256 units per layer, ReLU activations). Training hyperparameters are: learning rate $\eta = 3 \times 10^{-4}$, batch size $B = 2048$, GAE $\lambda = 0.95$, entropy coefficient $\beta = 0.01$. The agent was trained for 300,000 total timesteps (≈ 52 training episodes, each ≈ 5,800 timesteps) across 24 training days. All runs used a fixed random seed for reproducibility."

---

### 🟡 **Statistical Significance and Confidence Intervals**
**Missing:** Are metrics reported as mean ± std over multiple runs? How many seeds?

**Should clarify:**
- Do you report results from a single seed or averaged over N random seeds?
- Are confidence intervals or error bars included in figures/tables?
- Did you run each baseline multiple times with different random seeds?

**Where to add:** In "Training and Evaluation Protocol" or a new "Statistical Methodology" bullet.

**Suggested text:**
> "To account for stochasticity in worker/task arrivals and the DRL policy, all experiments were conducted over 5 independent random seeds. Results are reported as mean ± standard deviation across seeds. All baselines were evaluated on the same task/worker sequences to ensure a fair comparison."

---

### 🟡 **Fixed-Composite Tuning Details**
**Missing:** You mention "grid search on the training set" but don't specify:
- What was the search space? (e.g., λ₁ ∈ [0, 2], λ₂ ∈ [0, 0.5] with step size 0.1?)
- How was the best configuration chosen? (highest JFI? TAR? Scalarized multi-objective?)
- Was hyperparameter tuning done on the same 24 training days as the DRL, or a separate validation split?

**Where to add:** In the "Baselines" subsection.

**Suggested text:**
> "The Fixed-Composite baseline was tuned via grid search over λ₁ ∈ {0.0, 0.5, 1.0, 1.5, 2.0}, λ₂ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}, and γ ∈ {0.1, 0.3}. The configuration maximizing the average Jain's Fairness Index (JFI) on the 24 training days was selected for evaluation."

---

### 🟡 **Computational Cost and Scalability**
**Missing:** Training time, inference latency, memory footprint compared to baselines.

**Where to add:** New optional subsection "Computational Requirements" or mention in "Agent Configuration."

**Why it matters:** Readers want to know if your approach is practical. Baseline heuristics (greedy, LAF) run in O(n) or O(n²); if your DRL requires GPU and 24 hours of training, that's useful context.

---

### 🟡 **Warm-up State Statistics**
**Good:** You justify the 30-minute warmup. But **clarify expected state** after warmup:
> "After the 30-minute greedy warmup phase, the simulation reaches a steady-state backlog of approximately X tasks and an average worker idle time of Y minutes. The Governor then adjusts weights based on this non-empty system state."

---

### 🟡 **Evaluation Metrics Subsection — Keep It? YES, Absolutely.**

**My recommendation: KEEP THIS SECTION.**

**Rationale:**
1. **You defined a NEW problem:** The Bilateral Objective Asymmetry (BOA) framework is non-standard. Different papers operationalize "fairness" and "utilization" differently.
2. **Metrics are not interchangeable:** Jain's Fairness Index (JFI) vs. Gini coefficient vs. min-max fairness all measure different things. Readers need to know exactly which you're using.
3. **Standard practice in top-tier venues:** NeurIPS, ICML, KDD papers ALWAYS define metrics because reproducibility depends on it.
4. **Your metrics are heterogeneous:** You're measuring both fairness (JFI) and efficiency (TAR, wait time, backlog). The interplay matters.

**Improvement suggestion:**
Instead of just listing metrics, add a **short paragraph justifying each**:

```latex
\subsubsection{Evaluation Metrics}
\textbf{Jain's Fairness Index (JFI):} We employ JFI as our primary fairness measure 
because it is order-independent, dimensionless, and penalizes extreme inequality. 
The index is computed as the ratio of (sum of task counts)² to (n × sum of squared task counts), 
where n is the number of workers. Values range from 1/n (worst-case: one worker completes all tasks) 
to 1.0 (perfect fairness). This is superior to variance-based fairness measures in systems where 
workers have heterogeneous availabilities.

\textbf{Task Assignment Ratio (TAR):} The fraction of completed tasks relative to total arrivals. 
TAR directly reflects system capacity; lower TAR indicates tasks that expired or were not served, 
a hard constraint violation in spatial crowdsourcing.

\textbf{Average Wait Time:} The mean time from task release to assignment (not completion). 
This metric reflects the urgency experienced by requesters and is a primary quality-of-service metric 
in ride-hailing systems.

\textbf{Task Backlog:} Peak number of unassigned tasks in the queue at any point during the episode. 
High backlog indicates the system struggled with matching velocity; this is a secondary proxy for 
system congestion.
```

---

## 3. Minor Refinements

### 🟢 **References and Citations**
- Use `\autocite{CR-08}` consistently (you mixed `\cite` and `\autocite`)
- Ensure CR-08 and CR-11 are defined in your bibliography

### 🟢 **Notation Clarity**
- You use $r_j$ and $e_j$ for task release and expiry. Define these in a notation table if your paper is heavy on math, or at least say "where $r_j$ denotes the release time" on first use.

### 🟢 **Entity Modeling Wording**
- "Workers are implicitly removed from the candidate pool once they can no longer complete a trip before their shift deadline" — good, but you could be even more explicit: "Once a worker reaches their shift deadline, they are no longer eligible for new assignments and enter an 'offline' state."

---

## 4. Checklist: Is Section 5.1 Complete?

| Item | Status | Notes |
|------|--------|-------|
| Dataset description | ✅ | Specific, well-justified. |
| Regional context | ✅ | Coordinates, flat-earth approximation. |
| Sampling strategy | ✅ | Stratified temporal sampling preserves demand. |
| Worker modeling | ✅ | Start location, 8-hour shift, feasibility check. |
| Task modeling | ✅ | Release time, expiry, unit utility model. |
| Simulation architecture | ✅ | DES, event queue, temporal fidelity. |
| Train/eval split | ✅ | 80/20 day-level temporal split, no leakage. |
| Warmup justification | ✅ | 30-min greedy warmup is well-motivated. |
| Baselines | ✅ | 5 baselines, each justified. |
| **DRL hyperparameters** | 🔴 | **MISSING** — add PPO params, network arch, training epochs. |
| **Statistical methodology** | 🟡 | Number of seeds? Confidence intervals? |
| **Computational cost** | 🟡 | Training time, inference latency. |
| Evaluation metrics | ✅ | **KEEP IT** — add brief justifications per metric. |
| Reproducibility | 🟡 | Add hardware specs, random seed info. |

---

## 5. Summary and Recommendations

### Keep the Evaluation Metrics Subsection
**Decision: YES.** Add 1-2 sentences per metric explaining *why* you chose it and what it tells you about your system. This is standard in top venues.

### Before Submission
**Priority order:**
1. **Add DRL hyperparameters** (critical for reproducibility)
2. **Add statistical methodology** (# of seeds, confidence intervals)
3. **Add computational cost** (practical impact)
4. **Refine metric justifications** (polish)

### Current Strength
Your experimental setup is **stronger than most conference papers** in that you:
- Explicitly handle temporal leakage
- Use real spatiotemporal data with proper sampling
- Compare against diverse, well-motivated baselines
- Clearly describe simulation physics

The gaps are relatively minor—mainly documentation/transparency rather than methodological flaws.

---

## Example Revised Section Structure

```
5.1 Experimental Setup
  5.1.1 Dataset and Regional Realization
  5.1.2 Entity and Shift Modeling
  5.1.3 Simulation Physics and Constraints
  5.1.4 Agent Configuration                    [NEW: add DRL hyperparams]
  5.1.5 Training and Evaluation Protocol       [EXPANDED: add # seeds, CI details]
  5.1.6 Baselines                              [EXPANDED: clarify Fixed-Composite tuning]
  5.1.7 Evaluation Metrics                     [KEEP and expand with justifications]
  5.1.8 Computational Requirements             [NEW: add training time, latency]
```

This structure makes it clear to reviewers that every decision was intentional and reproducible.
