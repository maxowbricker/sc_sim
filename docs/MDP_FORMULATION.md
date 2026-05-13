# MDP Formulation — Section 4.3 Reference Document

This document provides all exact values, formulas, and justifications needed to write
Section 4.3 of the paper. All constants are extracted directly from `rl/gym_environment.py`,
`metrics/manager.py`, and `config.py` on the `conference-ready` branch.

---

## 1. The MDP Tuple

The Governor is formulated as a Markov Decision Process (MDP):

$$\mathcal{M} = \langle \mathcal{S},\; \mathcal{A},\; \mathcal{P},\; \mathcal{R},\; \gamma \rangle$$

| Symbol | Definition |
|--------|-----------|
| $\mathcal{S} \subseteq \mathbb{R}^{15}$ | Continuous 15-dimensional observation space |
| $\mathcal{A} \subseteq [-1, 1]^2$ | Symmetric continuous action space (network output) |
| $\mathcal{P}$ | Stochastic transition dynamics (driven by real spatiotemporal data) |
| $\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ | Δ-JFI composite reward signal |
| $\gamma = 0.95$ | Discount factor (from `best_hyperparameters.json`) |

The agent operates at a **5-minute decision interval** ($\Delta T = 5$ min), yielding
**96 decision steps** per 8-hour episode.

---

## 2. State Space $\mathcal{S}$

The state vector $\mathbf{s}_t \in \mathbb{R}^{15}$ captures the bilateral system's
**current health**, **momentum (one-step deltas)**, and **cyclical temporal context**.

### 2.1 Complete Feature Table

| Index | Symbol | Raw Source | Normalization | Description |
|-------|--------|-----------|---------------|-------------|
| 0 | $\phi_{\text{defer}}$ | `deferred_ratio` | None (natural [0,1]) | Fraction of all released tasks currently deferred (unassigned but not yet expired) |
| 1 | $\phi_{\text{avail}}$ | `worker_availability_ratio` | None (natural [0,1]) | Fraction of total workers currently available for assignment |
| 2 | $\phi_{\text{workers}}$ | `total_workers` | $\div\; 10{,}000$ | Normalised total worker count (divisor = `target_workers` from config) |
| 3 | $\phi_{\text{backlog}}$ | `backlog_peak` | $\div\; 200$ | Peak unassigned task queue depth this episode, normalised by reference scale |
| 4 | $J_t$ | `jfi` | None (natural [0,1]) | Jain's Fairness Index over all active workers at current time |
| 5 | $\Delta J_t$ | $J_t - J_{t-1}$ | $\div\; 0.05$ | Step-over-step JFI change (**Δ JFI momentum signal**) |
| 6 | $\bar{w}_t$ | `step_avg_wait` | $\div\; 2.0\text{ min}$ | Average task wait time in the current step window |
| 7 | $\Delta \bar{w}_t$ | $\bar{w}_t - \bar{w}_{t-1}$ | $\div\; 10.0\text{ min}$ | Step-over-step change in average wait time |
| 8 | $\Delta b_t$ | $b_t - b_{t-1}$ | $\div\; 30$ | Step-over-step change in peak backlog |
| 9 | $\Delta \lambda_t$ | $\lambda_t - \lambda_{t-1}$ | $\div\; 40.0\text{ tasks/min}$ | Step-over-step change in task arrival rate (demand velocity) |
| 10 | $d_{\text{mid}}$ | `is_midweek` | Binary {0, 1} | 1 if Tue–Thu, 0 otherwise |
| 11 | $d_{\text{mf}}$ | `is_mon_fri` | Binary {0, 1} | 1 if Mon or Fri, 0 otherwise |
| 12 | $d_{\text{wknd}}$ | `is_weekend` | Binary {0, 1} | 1 if Sat–Sun, 0 otherwise |
| 13 | $\tau_{\sin}$ | `time_sin` | Natural [-1,1] | $\sin(2\pi h / 24)$ — cyclical encoding of hour-of-day |
| 14 | $\tau_{\cos}$ | `time_cos` | Natural [-1,1] | $\cos(2\pi h / 24)$ — cyclical encoding of hour-of-day |

**Source:** `rl/gym_environment.py` → `_get_observation()`, lines 286–302.  
**Scaling constants:** `config.py` → `OBSERVATION_STATIC_SCALING`.

### 2.2 Design Rationale

The state is partitioned into three semantic groups:

**Group 1 — System-Level Snapshot (indices 0–4):**
Absolute levels giving the Governor a picture of current system health. The deferred ratio
and worker availability directly capture bilateral supply-demand balance. JFI ($J_t$) is
included as a level signal to prevent the agent from ignoring the accumulated fairness state.

**Group 2 — Momentum Signals (indices 5–9):**
Step-over-step deltas are the core credit-assignment mechanism. Without delta features,
the JFI signal suffers from a delayed attribution problem: assignments made at step $t$
complete 3–5 steps later, making it difficult for the policy to learn from absolute levels.
$\Delta J_t$ (index 5) provides an **immediately attributable** fairness signal.
All deltas are normalized to the $[-1, 1]$ range under typical operating conditions using
empirically calibrated divisors (p99 from composite Pareto sweep data).

**Group 3 — Cyclical Temporal Encoding (indices 10–14):**
Day-type flags (indices 10–12) allow the Governor to develop different policies for
workday rush hours versus weekend leisure patterns. Indices 13–14 encode hour-of-day
using **sine–cosine projection** to preserve the circular topology of time (midnight
is close to 11 pm, not far from it), a standard technique in time-series forecasting.

**Exclusion of last-action from state:**
Earlier versions included $(\lambda_1^{t-1}, \lambda_2^{t-1})$ in the observation vector
(17-dimensional). This was **removed** because it created a self-fulfilling collapse: once
$\lambda_1 \to 0$, the agent observed its own low value and learned that "low $\lambda_1$
is the norm," reinforcing the collapse. Removing these two features reduced the observation
to 15 dimensions and, combined with the symmetric action space (Section 4.4), resolved the
boundary-sticking problem.

---

## 3. Action Space $\mathcal{A}$

### 3.1 Raw Network Output

The PPO policy outputs a 2-dimensional continuous action vector:

$$\mathbf{a}_t = (a_1, a_2) \in [-1, 1]^2$$

This symmetric action space is enforced via `spaces.Box(low=-1.0, high=1.0, shape=(2,))`.

### 3.2 Physical Mapping

In `step()`, the raw network outputs are mapped to the Dispatcher's weight parameters:

$$\lambda_1^t = \text{clip}(a_1,\, -1, 1) + 1.0 \;\in\; [0.0,\; 2.0]$$

$$\lambda_2^t = \bigl(\text{clip}(a_2,\, -1, 1) + 1.0\bigr) \times 0.25 \;\in\; [0.0,\; 0.5]$$

$$\lambda_3 = 1.0 \quad (\text{fixed anchor})$$

**Source:** `rl/gym_environment.py` → `step()`, lines 218–219.

### 3.3 Symmetry Motivation and Boundary-Sticking Fix

Prior implementations used a direct `Box(low=0, high=2)` action space for $\lambda_1$. PPO's
Gaussian policy has support over $(-\infty, +\infty)$, but it is clipped at the box boundary.
When $\lambda_1 \to 0$, the gradient for "move away from zero" is suppressed because the
Gaussian density at the boundary accumulates probability mass — the so-called
**boundary-sticking problem**.

By centering the action space at zero, the natural initialisation point $a_1 = 0$ maps to
$\lambda_1 = 1.0$, which is also the Optuna-tuned optimal value for the static composite
baseline. This means the agent **starts at the best-known static operating point** and must
actively learn to deviate from it. Empirically, this change increased mean $\lambda_1$
from 0.133 (collapsed near zero) to 0.891 in 25k-step trials.

### 3.4 Fixed Anchor $\lambda_3 = 1.0$

The utility weight $\lambda_3$ (inverse pickup distance term) is held fixed at 1.0.
This is the **unit anchor** of the composite scoring function. Allowing the agent to
zero out $\lambda_3$ would produce absurd assignments (ignoring distance entirely) and would
make the action space non-identifiable with respect to scale. Fixing $\lambda_3 = 1.0$
defines the effective scale of the other two weights.

---

## 4. Reward Function $\mathcal{R}$

### 4.1 Design Philosophy

The reward function must solve two conflicting constraints:

1. **Credit assignment:** JFI is a simulation-wide rolling statistic. An assignment at step $t$
   only manifests in JFI after the task is completed 2–5 steps later.
2. **Multi-objective balance:** The Governor must simultaneously care about fairness (JFI),
   throughput (wait time / latency), and starvation prevention (task expirations).

### 4.2 Exact Formula

The reward at step $t$ is:

$$r_t = \frac{1}{5.0} \Bigl( \omega_F \cdot r_{\text{fairness}} + \omega_S \cdot r_{\text{starvation}} + \omega_L \cdot r_{\text{latency}} \Bigr)$$

where the three components are:

$$r_{\text{fairness}} = \Delta J_t \times 1000$$

$$r_{\text{latency}} = -\bar{w}_t \times 2.0$$

$$r_{\text{starvation}} = -E_{30} \times 0.5$$

And the outer weights are all 1.0 by default: $\omega_F = \omega_S = \omega_L = 1.0$
(controlled by `self.reward_weights`, default `[1.0, 1.0, 1.0]`).

**Source:** `rl/gym_environment.py` → `_calculate_reward()`, lines 306–338.

### 4.3 Component Definitions

| Component | Symbol | Definition | Scale factor | Signal type |
|-----------|--------|-----------|--------------|-------------|
| Fairness | $r_{\text{fairness}}$ | $\Delta J_t = J_t - J_{t-1}$ | $\times 1000$ | Step-attributable momentum |
| Latency | $r_{\text{latency}}$ | $-\bar{w}_t$ (avg wait in current step, minutes) | $\times 2.0$ | Absolute level penalty |
| Starvation | $r_{\text{starvation}}$ | $-E_{30}$ (tasks expired in trailing 30-min window) | $\times 0.5$ | Rolling count penalty |

### 4.4 The Δ JFI Momentum Signal

The choice of $\Delta J_t$ over $|J_t|$ is the key reward-engineering contribution.

**Why not absolute JFI?**
Absolute JFI $J_t$ at step $t$ reflects the cumulative history of all assignments since
the start of the day. A single step's action changes $J_t$ by at most $O(10^{-3})$.
This tiny, noisy signal is dominated by the latency penalty, which responds immediately
and provides $O(1)$ magnitude gradients. As a result, the agent ignores fairness
and collapses to a pure latency optimizer.

**Why Δ JFI works:**
$\Delta J_t = J_t - J_{t-1}$ isolates **what changed this step**. A positive $\Delta J_t$
means fairness improved relative to the previous step. The $\times 1000$ scaling ensures that
a typical improvement of +0.001 JFI yields a fairness reward of approximately +0.2, which is
on the same order of magnitude as the latency penalty. This scale parity allows the policy
gradient to receive balanced signal from both objectives.

**Credit assignment property:**
$\Delta J_t$ is computed in `step()` **before** `_get_observation()` updates `self.prev_jfi`.
This ordering ensures the delta is computed between the pre-action JFI and the post-step JFI,
correctly attributing the fairness change to the action taken at step $t$.

### 4.5 Latency and Starvation Components

**Latency ($r_{\text{latency}}$):**
$\bar{w}_t$ is the step-average wait time (time from task release to assignment, minutes)
over the current 5-minute window. This is an **absolute** level signal rather than a delta
because latency responds immediately to the current weights — a higher $\lambda_1$ now causes
more consideration of fairness vs. distance, which may increase or decrease wait time within
the same step.

**Starvation ($r_{\text{starvation}}$):**
$E_{30}$ is the count of tasks that **expired** in the trailing 30-minute window
(not just the current step). Using a rolling window rather than a per-step count provides a
smoother penalty signal and prevents reward hacking where the agent could momentarily reduce
expirations in a single step by flood-assigning.

### 4.6 Normalisation by 5.0

The entire reward is divided by 5.0. This maps the typical undiscounted episode return
($\approx$ 96 steps $\times$ few units per step) into the $[-30, +30]$ range that
is standard for PPO training stability.

---

## 5. PPO Hyperparameters

From `best_hyperparameters.json` (SHA `bb834b0e`, Optuna-tuned):

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Learning rate $\eta$ | $3 \times 10^{-4}$ | Adam optimizer |
| Steps per rollout $N$ | 2048 | Timesteps collected before each update |
| Batch size $B$ | 256 | Mini-batch size for gradient updates |
| Discount factor $\gamma$ | 0.95 | Horizon ≈ 20 steps ($1/(1-\gamma)$) |
| GAE $\lambda$ | 0.90 | Bias-variance trade-off for advantage estimation |
| Clip range $\epsilon$ | 0.2 | PPO trust-region clip |
| Entropy coefficient $\beta$ | 0.04 | Exploration regulariser |
| Value function coeff | 0.55 | Relative weight of value loss |
| Max gradient norm | 1.0 | Gradient clipping |
| Network architecture | Large [256, 256] | Two hidden layers, 256 units each, shared MLP |

**Training budget:** 300,000 timesteps ≈ 3,125 rollout batches ≈ 52 complete episodes.

**Parallelism:** 8 parallel `SubprocVecEnv` environments, each sampling a different day and
random drop-in start time, giving the agent diverse spatiotemporal experience.

---

## 6. Episode Structure (for Section 4.4 / Training Protocol)

```
Episode t:
  [0s – 1800s]  GREEDY WARMUP          Pure greedy assignment; populates backlog,
                                        heterogeneous idle times, non-zero JFI baseline
  [1800s – 30600s]  RL CONTROL PHASE   96 decision steps × 300s each
                    Step k:
                      1. Apply (λ1, λ2) to Dispatcher
                      2. Simulate 5 minutes of events (TASK_RELEASE, WORKER_RELEASE, etc.)
                      3. Compute ΔJFI, latency, starvation
                      4. Emit reward rt and next state st+1
```

**Random drop-in:** At the start of each episode, the simulation start time is drawn uniformly
from all valid windows in the selected training day. This means the agent never sees the exact
same demand pattern twice, acting as a form of data augmentation over the 24-day training corpus.

---

## 7. Key Technical Highlights for the Paper

| Contribution | Where to mention | What to say |
|---|---|---|
| **Δ JFI momentum signal** | Section 4.3.3 Reward | "Step-over-step JFI change resolves the credit assignment delay inherent in absolute JFI, enabling balanced fairness-latency gradient signals." |
| **Symmetric action space** | Section 4.3.2 Action | "Centering the action space at zero initializes the policy at the Optuna-optimal static operating point, eliminating boundary-sticking collapse." |
| **Exclusion of last-action** | Section 4.3.1 State | "Removing prior actions from the observation prevents self-fulfilling-prophecy collapse where observed near-zero λ₁ reinforces low fairness weight outputs." |
| **Cyclical temporal encoding** | Section 4.3.1 State | "Hour-of-day is encoded as sine–cosine projections (indices 13–14) preserving the circular topology of time." |
| **30-min greedy warmup** | Section 4.4 Training | "A greedy warmup initializes each episode with a non-empty, realistic system state, ensuring the Governor learns to govern an active bilateral market." |
| **Rolling 30-min starvation window** | Section 4.3.3 Reward | "The starvation penalty uses a trailing 30-minute window to smooth the expiration signal and prevent single-step reward hacking." |

---

## 8. LaTeX Skeleton for Section 4.3

```latex
\subsection{Markov Decision Process Formulation}
\label{sec:mdp}

We formalize the Governor's control problem as an MDP
$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$,
where the agent observes the bilateral system state every $\Delta T = 5$ minutes and outputs
continuous weight adjustments for the Dispatcher's composite scoring function.

\subsubsection{State Space $\mathcal{S}$}

The 15-dimensional observation vector $\mathbf{s}_t \in \mathbb{R}^{15}$ is partitioned into
three semantic groups: (i) system-level snapshot (indices 0–4), (ii) one-step momentum signals
(indices 5–9), and (iii) cyclical temporal context (indices 10–14). [Insert Table here]

Prior action $(\lambda_1^{t-1}, \lambda_2^{t-1})$ is \emph{excluded} from the state to
prevent self-fulfilling-prophecy collapse, a phenomenon observed in preliminary experiments
where the agent observed near-zero $\lambda_1$ and reinforced low fairness weights.

\subsubsection{Action Space $\mathcal{A}$}

The Governor outputs $\mathbf{a}_t \in [-1, 1]^2$ (symmetric Box space). To initialize the
policy at the known-optimal static operating point, raw outputs are mapped as:
\begin{align}
  \lambda_1^t &= a_1 + 1.0 \;\in\; [0.0,\, 2.0] \\
  \lambda_2^t &= (a_2 + 1.0) \times 0.25 \;\in\; [0.0,\, 0.5]
\end{align}
The utility weight is fixed as the anchor $\lambda_3 = 1.0$.

\subsubsection{Reward Function $\mathcal{R}$}

The reward signal must jointly address fairness and latency while resolving the
\emph{credit-assignment delay} inherent in JFI — a simulation-wide statistic that
responds to assignments made 2--5 decision steps earlier. We address this with a
momentum-based formulation:

\begin{equation}
  r_t = \frac{1}{5}\Bigl(
    \underbrace{\Delta J_t \cdot 1000}_{\text{fairness momentum}}
    \;-\; \underbrace{2\,\bar{w}_t}_{\text{latency}}
    \;-\; \underbrace{0.5\,E_{30}}_{\text{starvation}}
  \Bigr)
\end{equation}

where $\Delta J_t = J_t - J_{t-1}$ is the step-over-step change in Jain's Fairness Index,
$\bar{w}_t$ is the average task wait time (minutes) in the current step window, and $E_{30}$
is the count of tasks that expired in the trailing 30-minute window.
```
