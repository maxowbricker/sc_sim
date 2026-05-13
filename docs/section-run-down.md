# Paper — introduction & related work run-down

**Document purpose:** Scratchpad for **§1 Introduction** and **§2 Related work** — what each block should achieve, structural notes, and **paper IDs** in the "Papers to be referenced" lists (`CR-##`, `SA-##`, `TW-##`, `OT-##`, `TR-##`, …).

Each block uses: **Purpose** (your one-liner), **Rundown** (argument bullets), **Structure / notes** (subsubsections / traps), **Papers to be referenced** (one bullet per paper: `` `ID` (short name): point ``; leave blank colons or omit bullets until you add them).

---

## How this outline was chosen (so future-you remembers)

You had **two plausible storylines**:

1. **Fairness-first** (older sketch): §2.1 supply-side fairness → §2.2 demand/starvation → §2.3 RL → §2.4 metric gap.
2. **Funnel / world-first** (ICDM-style): establish the **bilateral online** problem *before* fairness, so reviewers cannot collapse your contribution to "FETA already did fairness."

**Resolution (current blueprint):** use **funnel + strategic critique** — start with the **world** (two-sided online bipartite matching in SC), then **equity** (fairness + **metric gap** / EWMA), then **reliability / starvation** (demand-side pillar), then **RL governance** (governor vs direct matchers). **§2.4 stays RL last** — it is the "cherry on top" that stitches the tripartite objective under non-stationarity.

**Order (agreed working plan):**

1. Bilateral context ("the world")
2. Fairness — supply-side / equity goal (+ real-time signaling)
3. Starvation / reliability — demand-side goal (contrast to passive "utility decay" framing)
4. RL in SC — "policy governor" as closure

**How this compares to "similar papers" (positioning notes, not section titles):**

| Paper | How they do related work | Your hinge |
|-------|--------------------------|------------|
| FATP / CR-08 | Task assignment → formulation → fairness & **timeliness** (incentives / finish faster) | Your **starvation** is **demand-centric waiting / abandonment**, not their timeliness story — say so once when you cite them. |
| LAF / CR-10 | Very small RW; ride-hailing assignment + fairness | You **must** situate them **after** you have defined the online bilateral world — you have *more* RL/fairness neighbors than they did. |
| F-Aware / CR-04 | Classic **funnel**: broad crowdsourcing → spatial SC → **fairness niche** | You can echo the funnel but add a layer your novelty needs: **static / symmetric-priority SC ≠ non-stationary city with asymmetric triggers**. |

**Socio-economic “why” layer (TW-##):** These sit **outside** pure algorithmics — use them so the math reads as **economically and industrially motivated** (good for ICDM).

| ID | Role | Where |
|----|------|--------|
| **TW-01** | **Throughput** — ride-hail SC needs extreme request rates; auction-style assignment is fast but **utilitarian** | §2.1 bilateral / event-driven story |
| **TW-02** | **Industry** — Lyft-style **online RL** dispatch is real | §2.4 RL governance; bridge **theory ↔ production** |
| **TW-03** | **Societal** — **Silent Labor** and **causal** earnings impact | **§1.3 Introduction** *and* §2.2.3 metric gap |

Meeting one-liner: *EWMA targets behaviorally validated **Silent Labor** ([TW-03]); the Governor extends **production** online RL dispatch ([TW-02]) with an explicit **fairness–starvation** control loop.*

---

## 1.1 Introduction — *context & motivation*

**Purpose**

*(Your one sentence: why spatial crowdsourcing assignment matters now — throughput, non-stationarity, stakeholder tension.)*

**Rundown**

- *(Optional hook: earnings **inequality** and **supply-side instability** over long horizons are an **economic** problem for the platform — not only a “nice-to-have” fairness norm.)*

**Structure / notes**

- **Opening stakes:** You may cite **`CR-16`** in the **first or second paragraph** to argue that cross-driver **earnings variance** threatens **long-run supply stability** (business-scale motivation, not only a math fix).

**Papers to be referenced:**

- `CR-16` (Long-term fairness in ride-hailing, 2024): **Economic stake** — earnings dispersion is not merely “unfair”; it undermines **platform supply resilience** over long horizons. *Pairs with the deeper fairness metric story in §2.4.*

---

## 1.2 Introduction — *problem, gaps, contributions*

**Purpose**

*(Your one sentence: what is broken today and what you claim to add — tripartite objective, BOA, governor.)*

**Rundown**

-

**Structure / notes**

- If you did not use **`CR-16`** in §1.1, repeat the **stakes** line here before listing contributions.

**Papers to be referenced:**

- `CR-16` (Long-term fairness in ride-hailing, 2024): *(optional repeat)* same **supply-side / long-horizon economic** framing as §1.1.

---

## 1.3 Introduction — *significance & “why” (Silent Labor hook)*

**Purpose**

*(Your one sentence: why **workers** and **platforms** should care — behavioral / economic grounding before the technical dive.)*

**Rundown**

-

**Structure / notes**

- Mirror the **metric story** in §2.2.3: you preview **Idle Labor / EWMA** here so §2 does not feel unmotivated.

**Papers to be referenced:**

- `TW-03` (Silent Labor Time):

---

## 2.1 Online bipartite matching in spatial crowdsourcing — *the bedrock*

**Purpose**

*(Your one sentence: define the stochastic two-sided online problem your simulator and governor sit inside.)*

**Rundown**

- **Focus:** two-sided **online** bipartite matching (TOBM) and how SC papers model **arrivals** (workers and tasks).
- **Punchline:** prior work accepts the **bilateral** setting, but often uses **symmetric priority** (similar treatment of task-release vs worker-availability). You need **asymmetric** control logic — different optimization pressure on **supply-side** vs **demand-side** triggers.
- **Throughput & economics ([TW-01]):** urban ride-hail–style SC requires **very high assignment throughput** (auction / online worker-side models are **fast** but **purely utilitarian**). That motivates keeping an **event-driven** assignment path while adding a **Governor** for **equity + starvation**, rather than defaulting to **batch-only** or **utility-only** designs.
- **Why §2.1:** grounds the **bilateral** substrate of the DRL stack before fairness / starvation / RL critiques.

**Structure / notes**

- **Open §2.1 with `OT-01` (first paragraph):** Hu & Zhou give the **optimal policy structure** for **dynamic two-sided markets** — a **Greedy Perfect-Pair**-style policy is often **structurally optimal**, which **justifies** your **Dispatcher** using a **greedy scoring** core as the matching engine (before you introduce asymmetry via the Governor).
- End §2.1 with one crisp contrast sentence you can reuse in §2.4: *symmetric trigger handling vs your asymmetric governor view*.
- Flow: **`OT-01`** (theoretical bedrock: greedy structure) → **`TW-01`** (throughput + utilitarian auction baseline) → `SA-01` (hardness / DRL license) → `SA-04` (TOBM benchmark) → `SA-05` / `SA-06` (arrival physics) → `CR-15` vs `CR-13` (symmetric bilateral vs preference trajectory) → `OT-04` (static multi-stage governor contrast).

**Papers to be referenced:**

- `OT-01` (Dynamic matching in a two-sided market — Hu & Zhou): **Theoretical bedrock** — characterizes **optimal policy structure** in two-sided dynamic matching; a **Greedy Perfect-Pair** policy is often **structurally optimal**. **Use:** opening paragraph of §2.1 to legitimize a **greedy** scoring core in the **Dispatcher** before layering **Governor** asymmetry.
- `TW-01` (Auction-SC — on-line task assignment in spatial crowdsourcing): Establishes **real-time throughput** demands of urban SC (e.g. **>** ~**10** requests **/ s** scale in ride-hailing narratives) and **decentralized / worker-side** auction-style scheduling as a **fast, utilitarian** baseline — supports choosing an **event-driven Dispatcher** for speed while **your Governor** adds **fairness** beyond pure utility.
- `SA-01` (GOMA): Defines the GOMA problem class and proves no deterministic algorithm can achieve a constant competitive ratio — provides the formal "academic license" to use DRL for priority discovery.
- `SA-04` (TOBM benchmark — spatial): Establishes the canonical unified definition of Two-sided Online Bipartite Matching (TOBM) in spatial data and demonstrates that no single static algorithm dominates all metrics — motivates the need for an adaptive governor to shift the system's operating point. (Your notes file: `SA-04_two-sided-online-bipartite-matching-in-spatial-data-experiments-and-analysis`.)
- `SA-05` / `SA-06` (OTA-TSA): Defines the stochastic arrival physics of bilateral markets via a Two-Stage Birth-Death Process — provides the theoretical grounding for your simulator's arrival model and justifies using Greedy as a robust baseline in small-batch regimes.
- `CR-13` (BDTA): Reframes bilateral matching as a trajectory-aware preference satisfaction problem — provides a contrast point to show that your work focuses on platform-level sustainability (Fairness + Starvation) rather than individual agent preferences.
- `CR-15` (Two-stage bilateral): Your primary structural ancestor; implements event-driven bilateral triggers but utilizes symmetric priority functions — this is the specific architectural simplification you fix with Bilateral Objective Asymmetry (BOA).
- `OT-04` (OTARP): Proposes a multi-stage framework that uses offline pre-predictions to guide online bilateral matching — serves as a "static-governor" predecessor that lacks the real-time weight-tuning required to handle non-stationary demand shifts.

---

## 2.2 Fairness-aware task assignment — *equity goal* (+ metric gap)

**Purpose**

*(Your one sentence: what "fair" means in prior SC assignment, and why their **metrics/instruments** are insufficient for closed-loop control.)*

**Rundown**

- **Focus:** balancing **utility vs worker earnings / exposure** — **FETA**, **F-Aware**, **LAF** and related assignment mechanics.
- **Differentiate:** **high-latency global optimization** (snapshot / batch fairness, often **O(W)**–**O(N³)** or **O(N·M)** matching) vs your **O(1) temporal signaling** (streaming EWMA-style instruments for closed-loop control).
- **Metric gap punchline:** **measurement matters as much as matching** — classical indices and WAF-style traces are wrong tool for city-scale, event-driven control unless they stream with **O(1)** work per event. Ground "why idle history" in **silent labor** ([TW-03]): causally tied to **earnings loss** and **churn**, not a convenience metric — frames the EWMA signal as **poverty / attrition prevention**, not only a technical choice.
- **Why §2.2 (not §2.1):** once the world is "truly online," you can say **FETA is too slow / wrong instrument** and **LAF is too rigid** without the reviewer short-circuiting to "fairness already solved."

**Structure / notes**

- **2.2.1 — Global optimization & batched frameworks:** **`CR-01`** (gradient dominance in multi-objective RL), FETA, F-Aware, LAF — *right idea (fairness is hard), often wrong math (too slow); naive **utility + fairness** sums fail in RL when scales differ*.
- **2.2.2 — Localized / online heuristics:** FATP — *right speed, snapshot-derived thresholds → wrong signal for true scalability*.
- **2.2.3 — Metric / signaling gap:** Jain's index (classic snapshot metric — **not** an `SA-##` paper), **TW-03**, EWMA — develop **TW-03** fully here; mirror the **Silent Labor** hook from **§1.3** so metric choice and literature align.

**Strategic note (“Killer 12” framing for prose)**

- **`CR-01`:** **utility rewards** can **dominate** **fairness** rewards by orders of magnitude → **gradient / signal starvation** on equity — motivates **min–max normalization** and a **DRL Governor** to stabilize multi-objective control (not a single naive weighted sum).
- **FETA / LAF:** fairness is hard; their math is often too slow for online triggers.
- **F-Aware / FATP:** heuristics are fast enough; their fairness / cap signals still lean on **global snapshots**.
- **Silent Labor ([TW-03]):** **causal** evidence linking **idle / relocation** to driver **earnings** and attrition (e.g. **~14.8%** earnings impact per km idle in their framing — verify exact number against your PDF). Positions EWMA-over-idle-history as **worker-side economic necessity**, not a parsimony-only trick.

**Papers to be referenced:**

- `CR-01` (Dynamic budgeted RL for fairness): Documents **gradient dominance** — **utility** terms often **overwhelm** **fairness** in naive **weighted-sum** RL, so the agent **ignores equity**. **Placement:** §2.2.1 (batched / global optimization narrative). **Your pivot:** **min–max normalization** + **Governor** to manage **numerical instability** and keep fairness visible in the learned policy.
- `CR-02` (Equity, Equality, Need): Grounds the work in distributive justice theory — fairness is multi-dimensional; justifies prioritizing a worker's **temporal experience** (idle history) over simple raw earnings.
- `CR-04` (F-Aware): Introduces the Local Assignment Ratio (LAR) as a greedy online fairness heuristic — an early **O(1)** incremental update story, but without the temporal memory and non-stationary adaptability of a DRL-governed signal.
- `CR-05` (FETA): Foundational many-to-many fairness in SC; **deserved share** via **O(N·M)** graph-matching snapshots — the **slow-but-fair** benchmark your stack seeks to scale past.
- `CR-06` (FGT/IEGT): Game-theoretic Nash-style equilibrium in worker payoffs — strong **long-run stability** anchor, but computationally unsuitable for **event-driven**, sub-second matching triggers.
- `CR-09` (FGTA): **Revenue-per-hour** fairness accounting for online duration — direct predecessor to normalizing equity by duration of **Silent Labor** intervals.
- `CR-10` (LAF): **Weighted Amortized Fairness (WAF)** over active time — temporally sensitive, but **tightly coupled** to the RL matching loop, unlike a portable, autonomous EWMA fairness signal.
- `CR-14` (Robust Fairness): Two-sided fairness via multi-population Genetic Algorithms — **cautionary complexity**; **~10 h** on NYC data reinforces the need for parsimonious **O(1)** signals.
- `TW-03` (Silent Labor Time): **Societal / economic anchor** — **causal** evidence that uncompensated **idle and relocation** (“**Silent Labor**”) deplete **gig** earnings and drive **attrition** (use their reported magnitudes, e.g. **~14.8%** earnings drop per km idle — **confirm in source**). **Primary “why”** for tracking **idle history** in an **EWMA** fairness signal: not only algorithmically cheap, but aligned with **documented** driver hardship. **Also cite in §1.3** so the contribution is motivated before §2.2.3.
- **Jain et al. (1998)** — *no `CR-##` / `SA-##` id; classic reference:* Jain, R., Chiu, D. M., & Hawe, W. (1998). A quantitative measure of fairness and discrimination for resource allocation in shared computer systems. CoRR, cs.NI/9809099. **Use in prose:** universal **O(W)** snapshot equity baseline in shared systems; your **O(1)** streaming EWMA aims to **track / approximate** that notion online.
- `CR-08` (FATP): Leading real-time heuristic with a **dynamic task-count cap**; critique reliance on a **global snapshot** to set the cap threshold **ĉ** — bottleneck to **true city-scale** scalability.

---

## 2.3 Task reliability and starvation prevention — *reliability goal*

**Purpose**

*(Your one sentence: demand-side **waiting / aging / abandonment** as a first-class objective, not an afterthought to worker fairness.)*

**Rundown**

- **Focus:** **MMD-SC**, **DSTA**, **Hybrid Q-learning waiting**, **AdaTaskRec**, **FATP** — delay, aging, and **task-side** sustainability.
- **Punchline:** much of the literature models delay as **passive utility decay** (utility drops as tasks age). In a **bilateral** market you want **active starvation pressure** so older / ignored tasks **gain** priority until matched — third leg of the tripartite objective.
- **AdaTaskRec as cousin:** your stack assumes **server-assigned tasks (SAT)** (platform assigns the match). **AdaTaskRec** lives in **worker-selected task (WST)** mode (recommend + worker chooses). Citing it shows reviewers that **ignoring tasks too long** is understood to hurt **platform sustainability** even in **decentralized recommendation** settings — starvation is not a SAT-only concern.
- **`CR-08` contrast:** timeliness as **passive decay** risks **orphaning** older tasks; your **active starvation signal** **increases** priority until assignment.

**Structure / notes**

- One paragraph separating **utility decay / timeliness** (e.g. FATP) from **starvation / worst-case delay / active incentives** (MMD-SC, DSTA, your **S**) avoids reviewer confusion.

**Strategic note (“logic trap” for prose)**

1. **Goal:** reduce worst-case / platform-visible delay (**MMD-SC**).
2. **Standard move:** let task utility **rot** or **decay** with age (**FATP**).
3. **Flaw:** decay means the task **looks less valuable** as it waits → less reason to ever match it → abandonment / orphaning.
4. **Your synthesis:** **extra-reward / ignored-task** intuition (**AdaTaskRec**, WST) + **logarithmic aging** math (**DSTA**) → **active incentive** **S** (SAT setting) so starvation is the signal that **prevents** silent abandonment.

**Papers to be referenced:**

- `CR-11` (MMD-SC): Foundational **min–max delay** framing — reduces worst-case wait but **does not** balance against **worker fairness** mechanisms.
- `OT-03` (DSTA): Time-dependent **urgency** in the utility; **logarithmic transform** for task aging — **direct mathematical inspiration** for starvation signal **S**.
- `SA-08` (Hybrid Scenarios): **Q-Learning** for **optimal waiting** before re-matching failed assignments — **wait-aware**, **reactive** starvation; yours treats starvation as a **proactive scoring priority** in the assignment objective.
- `CR-14` (Robust Fairness): **Genetic algorithm** two-sided fairness — empirical **non-convex** multi-objective trade-offs; **high latency** reinforces need for **O(1)** starvation logic.
- `AdaTaskRec` (AdaTaskRec — Zhao et al., 2023): **Challenge III** (fairness to **task requesters**) in **worker-selection** / recommendation: **extra rewards** for repeatedly **ignored** tasks — starvation prevention as **platform sustainability** even under **WST**; **cousin** to your **SAT** governor + **S**.
- `CR-08` (FATP): **Contrast** — timeliness as **passive utility decay** with age; risks **orphaning** older tasks; your **active starvation signal** **raises** priority until matched.

---

## 2.4 Reinforcement learning for platform governance — *the brain*

**Purpose**

*(Your one sentence: why **RL** is the right closure for **tripartite**, **non-stationary** SC — and why **direct matching policies** are the wrong abstraction.)*

**Rundown**

- **Focus:** **TW-02** (industrial online RL), **long-horizon fairness** peer (**`CR-16`**), **direct-matcher / symmetry** rival (**`CR-07`** FMRL), **trigger-timing** batch rivals (**SA-02**, **SA-07**), **threshold** RL (**SA-03**), **matcher-centric** baselines (**CR-10**, **CR-12**), **hierarchical Governor ally** (**TR-04** ARDE).
- **Punchline:** much prior RL acts as **direct matchers** (choose workers / edges) → **|W| × |T|** blow-up. You align with a **policy governor** — **meta-adaptive weighting** of tripartite objectives atop a fast assignment stack — as the scalable, interpretable integrator under **non-stationary** urban dynamics.
- **Tie-back:** §2.1 asymmetric triggers + §2.2–2.3 objectives are the operational **conflict**; §2.4 is the **technical climax**: RL as **governance**, not **matching**.
- **Industry bridge ([TW-02]):** production dispatch already proves **value-based online RL** at scale; you **extend** that lineage with explicit **fairness** and **starvation** feedback — the loop industrial stacks typically **omit**.

**Structure / notes**

- Keep **RL last** — related work culminates in **your architectural commitment** (simulator + governor), not in fairness definitions alone.
- **Pitch arc (meeting one-liner):** industrial reality (**TW-02**) → field moved to **long-horizon** fairness (**`CR-16`**) → **direct matcher + symmetric** FMRL rival (**`CR-07`**) → scalability wall (**CR-10**) → **hierarchical legitimacy** (**`TR-04`**) → **asymmetric** governor vs **symmetric** tripartite baselines (**CR-12**).
- **Reading order (suggested):** **`TW-02`** → **`CR-16`** (temporal / sustainability turn) → **`CR-07`** (FMRL symmetry critique) → **`SA-02` / `SA-07`** → **`SA-03`** → **`CR-10` / `CR-12`** → **`TR-04`** (close with architectural defense).

**Strategic note (“technical climax” for prose)**

1. **Industrial:** Platforms (**TW-02**) already run online RL dispatch — RL is not exotic; **your** contribution is closing **fairness + starvation** with the same ethos.
2. **Scalability:** Seminal fairness RL (**CR-10**) **does matching** → **|W| × |T|** complexity; governor **does not**.
3. **Structure:** **ARDE** (**TR-04**) motivates **hierarchical** governance and **high-level** weight tuning (**b₁, b₂, b₃** / λ); you differ in **timescale** and **matching substrate** (see **TR-04** bullet).
4. **Moat:** **MSFTA** (**CR-12**) is **symmetric** (task / worker arrivals treated alike); you claim **asymmetric** priority control under bilateral triggers.
5. **Long-horizon peer:** **`CR-16`** shifts discourse from **instant** fairness to **long-term sustainability** (e.g. **MLP** look-ahead **~1 week**) — **pivot:** while that addresses stability forecasting-wise, **MOMAQL** / tabular limits **scalability** in **high-frequency** cities; **your** PPO Governor + event stack targets that gap (**wording TBD** vs their exact algorithm).
6. **Symmetry rival:** **`CR-07`** (**FMRL**, fairness-aware dynamic ride-hailing, **2024**) — **direct** applicational neighbor; **critique:** **trigger symmetry** (same fairness logic every **~2 s** batch). You: **event-driven Bilateral Objective BOA** is **physically** and **computationally** closer to **asymmetric** real operations.

**Papers to be referenced:**

- `TW-02` (Lyft OSV): Industrial **existence proof** for deploying **value-based online RL** in production dispatch; you adopt their focus on **long-horizon driver-side returns** but **extend** the stack to include **explicit fairness** and **starvation** control loops.
- `SA-02` (HLAP): **Dual-attention DQN** that adaptively partitions task streams into batches — **trigger-timing** rival: learning **when** to match vs your focus on **how** to prioritize via **adaptive weighting**.
- `SA-07` (DTAF-PAB): **GRU-DRL** pipeline for adaptive batching; reinforces **non-stationarity** awareness but relies on **symmetric**, **benefit-only** optimization within batches.
- `SA-03` (PPO-TA): Shows **PPO** can learn **match-quality thresholds** in **dynamic bilateral** markets; you **extend** from a **single global threshold** to **meta-adaptive weighting** across **three conflicting objectives**.
- `CR-16` (Long-term fairness in ride-hailing, 2024): **Long-horizon ally** — community shift from **myopic** fairness to **sustainability**; uses **MLP** forecasting **~one week** ahead. **Pivot in prose:** *While **CR-16** targets long-term stability via forecasting, reliance on **MOMAQL** / tabular structure limits **scalability** under **high-frequency** urban matching; we address this via …* (**finalize** contrast once methods are locked).
- `CR-07` (Fairness-aware dynamic ride-hailing — **FMRL**, 2024): **Direct matcher / symmetry rival** — strong ride-hail **fairness RL** peer applying the **same** fairness logic on every **~2 s** batch (**trigger symmetry**). You: **event-driven BOA** — **asymmetric** treatment of task- vs worker-initiated triggers; **more efficient** and **operationally faithful** to bilateral **physics**.
- `CR-10` (LAF): Seminal **RL-driven fairness** in ride-hailing; critique its **direct matcher**: **|W| × |T|** action-space blow-up → propose **policy governor** as more **scalable** and **interpretable**.
- `CR-12` (MSFTA): **Hybrid Ant-Q** explicitly targets the **tripartite** trade-off — strong multi-objective baseline but **symmetric**, **batch-synchronous** triggers; contrast with **asymmetric** governor prioritization.
- `TR-04` (ARDE — dual-layer evolution, 2026): **Governor ally / primary architectural defense** — legitimizes **two-tier** **platform regulation ≠ matching execution**. **Prose template:** *Following the hierarchical intuition of **ARDE** [**TR-04**], we separate **platform-level** regulation from **matching execution** …* Then **contrast** timescale / substrate (**sub-minute** bipartite **spatiotemporal** SC vs their **daily** strategy / evolution framing — keep aligned with your notes).

---

## Block template (copy for new sections)

```markdown
## 1.x or 2.x Title — *short tag*

**Purpose**

*(Your one sentence.)*

**Rundown**

-

**Structure / notes**

-

**Papers to be referenced:**

- ``:
```

---

## ID checklist

Fill the right column as you finalize the bibliography. Confirmed IDs already appear in the sections above; *(assign ID)* rows are still placeholders.

**SA numbering:** `SA-03` = **PPO-TA** (DDBM + PPO/PPG thresholds — §2.4; your `SA-03_ppo-ta-...` notes). `SA-04` = TOBM spatial benchmark (`SA-04_...` notes). Jain's index is **not** an `SA-##` entry.

**TW numbering (socio-economic / industry):** `TW-01` throughput + utilitarian auctions (§2.1); `TW-02` production RL dispatch (§2.4); `TW-03` Silent Labor + **§1.3** + §2.2.3.

| Work | ID | Section(s) | Role / note |
|------|----|------------|-------------|
| Auction-SC (on-line SC assignment) | `TW-01` | 2.1 | High **throughput** urban SC; fast **auction** baselines — **utilitarian**; motivates **event-driven** speed + **Governor** for fairness |
| GOMA (two-sided theory) | `SA-01` | 2.1 | No deterministic constant competitive ratio → formal license for DRL priority discovery |
| HLAP (batch timing RL) | `SA-02` | 2.4 | Dual-attention DQN batch partitions — *when* to match; you: *how* to prioritize (**adaptive weighting**) |
| PPO-TA (DDBM + PPO thresholds) | `SA-03` | 2.4 | PPO thresholds in DDBM → you: **meta-adaptive** weighting of **three** objectives (not one global threshold) |
| TOBM benchmark (spatial data) | `SA-04` | 2.1 | Unified TOBM definition + empirical “no algorithm dominates”; symmetric triggers, no fairness metric — your gap |
| OTA-TSA (stochastic) | `SA-05` / `SA-06` | 2.1 | Two-Stage Birth-Death arrival physics; Greedy baseline in small-batch regimes |
| DTAF-PAB (batch timing RL) | `SA-07` | 2.4 | GRU-DRL batching; non-stationarity — **symmetric benefit-only** batches (contrast tripartite governor) |
| Hybrid Scenarios (Q-Learning) | `SA-08` | 2.3 | Q-Learning waiting / re-match — reactive wait-aware predecessor vs proactive **S** |
| Dynamic budgeted RL (fairness) | `CR-01` | 2.2.1 | **Gradient dominance** (utility ≫ fairness) → min–max norm + **Governor** |
| Equity / Equality / Need | `CR-02` | 2.2 | Distributive justice; temporal experience vs raw earnings |
| F-Aware | `CR-04` | 2.2 | LAR greedy O(1) updates; lacks DRL temporal memory |
| FETA | `CR-05` | 2.2 | O(N·M) deserved-share snapshots; slow-but-fair benchmark |
| FGT / IEGT (game-theoretic) | `CR-06` | 2.2 | Nash equilibrium stability; too heavy for sub-second triggers |
| FMRL (fairness-aware dynamic RH, 2024) | `CR-07` | 2.4 | **Symmetry** rival — same fairness logic every **~2 s** batch; you: **event BOA** |
| FATP | `CR-08` | 2.2, 2.3 | §2.2: cap ĉ snapshot; §2.3: passive decay vs active starvation — orphaning risk |
| FGTA (revenue-per-hour) | `CR-09` | 2.2 | Duration-normalized fairness; precedes Silent Labor framing |
| LAF | `CR-10` | 2.2, 2.4 | §2.2: WAF vs portable EWMA; §2.4: **direct matcher** (worker×task-scale actions) vs **governor** |
| MMD-SC | `CR-11` | 2.3 | Min–max delay objective; no fairness balance |
| MSFTA | `CR-12` | 2.4 | Hybrid **Ant-Q** tripartite trade-off; **symmetric batch-synchronous** triggers — contrast **asymmetric** governor |
| BDTA (bilateral + preferences) | `CR-13` | 2.1 | Trajectory-aware preference satisfaction; contrast platform sustainability (fairness + starvation) |
| Robust Fairness (GA-based) | `CR-14` | 2.2, 2.3 | Two-sided GA fairness; non-convex trade-offs; ~10 h NYC → need O(1) signals |
| Two-stage bilateral | `CR-15` | 2.1 | Structural ancestor: symmetric priorities under bilateral triggers → you fix with BOA |
| Long-term fairness ride-hailing (2024) | `CR-16` | **1.1 / 1.2**, 2.4 | Intro: **earnings variance → supply stability**; RW: long-horizon + MLP vs **MOMAQL** scalability pivot |
| Lyft OSV (production RL) | `TW-02` | 2.4 | Value-based **online RL** dispatch; **long-horizon** driver returns → you add **fairness + starvation** loops |
| Silent Labor | `TW-03` | **1.3**, 2.2 | Causal **idle → earnings / attrition**; hook EWMA in intro + metric gap (**~14.8%** — verify in PDF) |
| ARDE (hierarchical RL) | `TR-04` | 2.4 | Dual-layer hierarchy; you: **sub-minute** bipartite **spatiotemporal** constraints vs **daily** strategy diversity |
| Hu & Zhou dynamic two-sided matching | `OT-01` | 2.1 | **Greedy perfect-pair** structural optimality → justifies **greedy Dispatcher** core |
| DSTA (log-urgency) | `OT-03` | 2.3 | Log aging / urgency → inspiration for starvation signal **S** |
| OTARP (multi-stage, offline preds) | `OT-04` | 2.1 | Static-governor predecessor; no real-time weight tuning for non-stationary demand |
| AdaTaskRec (Zhao et al. 2023) | `OT-05` | 2.3 | WST / recommendation; extra rewards for ignored tasks — starvation & platform sustainability; cousin to SAT |
| Jain et al. (1998), fairness index | *(none — bib string only)* | 2.2 | `cs.NI/9809099`; O(W) snapshot vs streaming O(1) EWMA |
