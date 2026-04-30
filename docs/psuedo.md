# Pseudocode — Bilateral Dispatcher ($E_t$ and $E_w$)

The dispatcher operates under two asymmetric event triggers. A **task-initiated event** $E_t$ fires when a batch $\mathcal{T}^{\text{new}}$ of tasks is released and idle workers exist; a **worker-initiated event** $E_w$ fires when a worker completes a task or enters the system and the deferred pool $\mathcal{T}^{\text{def}}$ is non-empty. Both paths share a common two-phase structure: a fast spatial gate (Phase 1) that narrows the search space using an approximate nearest-neighbour index, followed by full feasibility and scoring evaluation (Phase 2).

---

## Sub-procedure: Spatial Candidate Generation

Both algorithms delegate Phase 1 to the same ANN routine, but query **different indices** — workers for $E_t$, deferred tasks for $E_w$.

``` 
Procedure  ANN-QUERY(index, ℓ_query, k)

  Input : spatial index over a set of entities,
          query location ℓ_query, neighbourhood size k
  Output: C — up to k nearest entities by location

  1:  C ← grid-bucket lookup centred on ℓ_query
  2:  return top-k entries of C by Manhattan distance to ℓ_query
```

> This is an $O(k)$ approximate scan over a pre-built grid; no scoring or feasibility check occurs here.

---

## Algorithm 1: Task-Initiated Assignment ($E_t$)

Triggered when task $t_j$ is released and $\mathcal{W} \neq \emptyset$.  
Objective: find the best **worker** for each incoming task, favouring **underserved workers** and **spatial efficiency**.

**Input:**
- $\mathcal{T}^{\text{new}}$ — batch of newly released tasks
- $\mathcal{W}$ — available workers (maintained in worker spatial index $\mathcal{I}_\mathcal{W}$)
- $\mathcal{T}^{\text{def}}$ — deferred task pool
- $\mathbf{b} = [b_1,\, b_2,\, b_3]^\top$ — weight vector from DRL governor
- $k$, $\theta$, $\gamma$ — neighbourhood size, soft threshold, EWMA factor

**Output:** Assignment set $\mathcal{A}$; updated $\mathcal{T}^{\text{def}}$

```
Algorithm 1  Task-Initiated Assignment  (E_t)

 1:  A ← ∅
 2:  for each t_j ∈ T^new do
 3:
 4:      ▷ Phase 1 — Spatial Candidate Generation
 5:      C(t_j) ← ANN-QUERY(I_W, ℓ_j, k)                                    ▷ Workers near t_j
 6:
 7:      ▷ Phase 2 — Feasibility Filtering and Scoring
 8:      best_w ← ∅ ;  best_f ← −∞
 9:      for each w_i ∈ C(t_j) do
10:
11:          d_pick ← dist(ℓ_i, ℓ_j)                                          ▷ Pickup distance (km)
12:          d_drop ← dist(ℓ_j, ℓ_j^dest)                                     ▷ Drop distance (km)
13:
14:          ▷ Hard feasibility constraints
15:          if  now + (d_pick / v) > τ_j^exp                then continue    ▷ Task expires before pickup
16:          if  now + ((d_pick + d_drop) / v) > τ_i^dl      then continue    ▷ Worker misses own deadline
17:
18:          ▷ Objective signals
19:          Δt_idle    ← now − max(τ_i^last, τ_i^on)
20:          F(w_i)     ← (1 − γ) · Δt_idle + γ · F^(k−1)(w_i)              ▷ EWMA fairness (Eq. 2)
21:          U(w_i,t_j) ← 1 / (1 + d_pick)                                   ▷ Spatial utility  (Eq. 3)
22:
23:          ▷ Task-side score: ranks workers by fairness + utility
24:          score(w_i) ← b_1 · F(w_i) + b_3 · U(w_i, t_j)                  ▷ f_task           (Eq. 1)
25:          if score(w_i) > best_f  then
26:              best_f ← score(w_i) ;  best_w ← w_i
27:          end if
28:      end for
29:
30:      ▷ Task-level starvation offset (constant across candidates — does not affect worker ranking)
31:      S(t_j)  ← log(1 + (now − τ_j^rel))                                  ▷ Starvation signal (Eq. 4)
32:      f_task  ← best_f + b_2 · S(t_j)
33:
34:      ▷ Assignment or deferral
35:      if best_w ≠ ∅  and  f_task ≥ θ  then
36:          τ_j^start  ← now + (d_pick / v)
37:          τ_j^finish ← τ_j^start + (d_drop / v)
38:          best_w.F^(k) ← F(best_w)                                         ▷ Persist EWMA update
39:          A ← A ∪ {(t_j, best_w)} ;  remove best_w from W
40:      else
41:          T^def ← T^def ∪ {t_j}                                            ▷ Defer; schedule expiry
42:      end if
43:  end for
44:  return A, T^def
```

---

## Algorithm 2: Worker-Initiated Assignment ($E_w$)

Triggered when worker $w_i$ becomes free (task completion or worker release) and $\mathcal{T}^{\text{def}} \neq \emptyset$.  
Objective: find the best **deferred task** for the newly available worker, favouring **starving tasks** and **spatial efficiency**. Fairness is not used to rank tasks; it enters once as a worker-level addend on the final score.

**Input:**
- $w_i$ — newly available worker at location $\ell_i$
- $\mathcal{T}^{\text{def}}$ — deferred task pool (maintained in task spatial index $\mathcal{I}_{\mathcal{T}^{\text{def}}}$)
- $\mathcal{W}$ — available worker pool
- $\mathbf{b} = [b_1,\, b_2,\, b_3]^\top$ — weight vector from DRL governor
- $k$, $\theta$, $\gamma$ — neighbourhood size, soft threshold, EWMA factor

**Output:** Assignment $(t^*, w_i)$ or $\emptyset$; updated $\mathcal{T}^{\text{def}}$

```
Algorithm 2  Worker-Initiated Assignment  (E_w)

 1:  if T^def = ∅  then  return ∅  end if
 2:
 3:      ▷ Phase 1 — Spatial Candidate Generation
 4:      C(w_i) ← ANN-QUERY(I_{T^def}, ℓ_i, k)                              ▷ Deferred tasks near w_i
 5:
 6:      ▷ Phase 2 — Feasibility Filtering and Scoring
 7:      best_t ← ∅ ;  best_s ← −∞
 8:      for each t_j ∈ C(w_i) do
 9:
10:          d_pick ← dist(ℓ_i, ℓ_j)                                          ▷ Pickup distance (km)
11:          d_drop ← dist(ℓ_j, ℓ_j^dest)                                     ▷ Drop distance (km)
12:
13:          ▷ Hard feasibility constraints
14:          if  now + (d_pick / v) > τ_j^exp                then continue    ▷ Task expires before pickup
15:          if  now + ((d_pick + d_drop) / v) > τ_i^dl      then continue    ▷ Worker misses own deadline
16:
17:          ▷ Objective signals
18:          S(t_j)     ← log(1 + (now − τ_j^rel))                           ▷ Starvation signal (Eq. 4)
19:          U(w_i,t_j) ← 1 / (1 + d_pick)                                   ▷ Spatial utility   (Eq. 3)
20:
21:          ▷ Worker-side score: ranks tasks by starvation + utility
22:          score(t_j) ← b_2 · S(t_j) + b_3 · U(w_i, t_j)                  ▷ f_worker          (Eq. 5)
23:          if score(t_j) > best_s  then
24:              best_s ← score(t_j) ;  best_t ← t_j
25:          end if
26:      end for
27:
28:      if best_t = ∅  then  return ∅  end if
29:
30:      ▷ Fairness addend — applied once to winner, not used in task ranking
31:      Δt_idle  ← now − max(τ_i^last, τ_i^on)
32:      F(w_i)   ← (1 − γ) · Δt_idle + γ · F^(k−1)(w_i)                   ▷ EWMA fairness     (Eq. 2)
33:      f_worker ← best_s + b_1 · F(w_i)
34:
35:      ▷ Assignment or abstain
36:      if f_worker ≥ θ  then
37:          d_pick_final ← dist(ℓ_i, ℓ_{best_t})
38:          d_drop_final ← dist(ℓ_{best_t}, ℓ_{best_t}^dest)
39:          τ_{best_t}^start  ← now + (d_pick_final / v)
40:          τ_{best_t}^finish ← τ_{best_t}^start + (d_drop_final / v)
41:          w_i.F^(k) ← F(w_i)                                               ▷ Persist EWMA update
42:          T^def ← T^def \ {best_t}
43:          return (best_t, w_i)
44:      else
45:          return ∅                                                           ▷ No suitable task; worker stays idle
46:      end if
```

---

## Notation Summary

| Symbol | Meaning |
|--------|---------|
| $E_t$, $E_w$ | Task-initiated / worker-initiated event trigger |
| $t_j \in \mathcal{T}$ | Task: pickup $\ell_j$, destination $\ell_j^{\text{dest}}$, release $\tau_j^{\text{rel}}$, expiry $\tau_j^{\text{exp}}$ |
| $w_i \in \mathcal{W}$ | Worker: location $\ell_i$, available since $\tau_i^{\text{on}}$, deadline $\tau_i^{\text{dl}}$, last active $\tau_i^{\text{last}}$ |
| $\mathcal{T}^{\text{def}}$ | Deferred task pool; re-evaluated on every $E_w$ |
| $\mathcal{I}_\mathcal{W}$, $\mathcal{I}_{\mathcal{T}^{\text{def}}}$ | Separate spatial grid indices for workers and deferred tasks |
| $\mathcal{C}(t_j)$, $\mathcal{C}(w_i)$ | Phase 1 candidate sets ($k$-nearest by location) |
| $F(w_i)$ | EWMA fairness: $(1-\gamma)\,\Delta t_{\text{idle}} + \gamma\, F^{(k-1)}(w_i)$ |
| $U(w_i, t_j)$ | Spatial utility: $1\,/\,(1 + \mathrm{dist}(\ell_i, \ell_j))$ |
| $S(t_j)$ | Starvation: $\log(1 + (t_{\text{now}} - \tau_j^{\text{rel}}))$ |
| $f_{\text{task}}(w_i, t_j)$ | $E_t$ scorer: $b_1\,F(w_i) + b_3\,U(w_i, t_j)$ (Eq. 1) |
| $f_{\text{worker}}(w_i, t_j)$ | $E_w$ scorer: $b_2\,S(t_j) + b_3\,U(w_i, t_j)$ (Eq. 5) |
| $\mathbf{b} = [b_1, b_2, b_3]^\top$ | DRL governor weights; $b_3$ fixed at $1.0$ |
| $\theta$ | Soft assignment threshold |
| $v$ | Average worker travel speed (km/h) |
| $\gamma$ | EWMA smoothing factor $\in [0, 1]$ |

---

## Structural Asymmetry Between $E_t$ and $E_w$

| | $E_t$ — Task-Initiated | $E_w$ — Worker-Initiated |
|---|---|---|
| **Phase 1 index queried** | Worker index $\mathcal{I}_\mathcal{W}$ (by task location) | Deferred task index $\mathcal{I}_{\mathcal{T}^{\text{def}}}$ (by worker location) |
| **Phase 2 candidates** | Workers $w_i \in \mathcal{C}(t_j)$ | Deferred tasks $t_j \in \mathcal{C}(w_i)$ |
| **Inner-loop signals** | $F(w_i)$, $U(w_i, t_j)$ — fairness ranks workers | $S(t_j)$, $U(w_i, t_j)$ — starvation ranks tasks |
| **Fairness role** | Per-candidate (drives worker selection) | Post-selection addend on winner only |
| **Starvation role** | Per-task constant offset (after worker ranking) | Per-candidate (drives task selection) |
| **On threshold fail** | $t_j \to \mathcal{T}^{\text{def}}$ | Worker remains idle; $\mathcal{T}^{\text{def}}$ unchanged |
 