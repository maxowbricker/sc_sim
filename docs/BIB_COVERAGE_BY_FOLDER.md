# BibTeX coverage audit (folders [1]–[6] vs `references.bib`)

**Sources (read-only):**

- Paper library: `/Users/maxapple/Documents/All/Career/Fair Task Allocation in Spatial Crowdsourcing/ICDM/papers of relevance/`
- BibTeX: `/Users/maxapple/Documents/All/Career/Fair Task Allocation in Spatial Crowdsourcing/ICDM/papers of relevance/references.bib`
- Normalization: DOIs compared case-insensitively after stripping a leading `https://doi.org/` (or `http://doi.org/`). Several bib entries store DOI with the URL prefix — treat as equivalent when the tail matches.

**Scope note:** Only folder **`[1] Critical Rivals`** has per-paper metadata in `notes/*.yml`. Folders **`[2]`–`[6]`** use `notes/*.md` only (no paper-level YAML). For those rows, the **title** is taken from the markdown H1 heading (or the obvious PDF title in `[6]`). There is **no** `LM-##` encoded-ID series; `[6] Last minute` holds ad hoc PDFs only.

**Bib hygiene:** `references.bib` contains **duplicate** `@article{CR-15,` entries (one IEEE TSC full citation, one placeholder “Zhang et al.”). Resolve before relying on BibTeX parsers that assume unique keys.

---

## [1] Critical Rivals

**`notes/*.yml` (paper-level):** `CR-01` … `CR-16`, plus `CR-TRIAL` (duplicate FATP metadata vs `CR-03`). Excluded: `index.yml` (folder manifest only).

**Loose PDFs (root):** none — each root PDF matches a `pdf_original_filename` in a sibling `notes/CR-##_*.yml`.

| ID | Slug / title (from YAML) | Match status | Bib key or MISSING | Notes |
|----|-----------------------------|--------------|-------------------|--------|
| CR-01 | Dynamic Budgeted Reinforcement Learning for Fairness in Spatial-Temporal Resource Allocation | Missing from bib | **MISSING** | No DOI in YAML; no matching title in bib. |
| CR-02 | Equity, Equality, and Need: Digital Twin Approach for Fairness-Aware Task Assignment of Heterogeneous Crowdsourced Logistics | Missing from bib | **MISSING** | No DOI in YAML; no matching title in bib. |
| CR-03 | FATP: Fairness-Aware Task Planning in Spatial Crowdsourcing | Missing from bib | **MISSING** this is basically the same paper as CR-08, treating it as the same for now. (TMC “An Online Fairness-Aware Task Planning Approach…”)
| CR-04 | Fair Task Allocation in Crowdsourced Delivery | In bib + linked | `CR-04` | Title + `doi=10.1109/TSC.2018.2854866`. |
| CR-05 | Fair Task Assignment in Spatial Crowdsourcing | In bib + linked | `CR-05` | Title alignment; bib entry has no DOI field. |
| CR-06 | Fairness-aware Task Assignment in Spatial Crowdsourcing: Game-Theoretic Approaches | In bib, ID not obvious | `Game-FairnessAware-2021` | Suggested linkage: encode ID **CR-06** → key `Game-FairnessAware-2021` (`doi=10.1109/ICDE51399.2021.00030`). |
| CR-07 | Fairness-Aware Dynamic Ride-Hailing Matching Based on Reinforcement Learning | Missing from bib | **MISSING** | No DOI in YAML; no confident title hit in bib. |
| CR-08 | An Online Fairness-Aware Task Planning Approach for Spatial Crowdsourcing | In bib + linked | `CR-08` | Key equals folder ID; `doi=10.1109/TMC.2022.3229112`. |
| CR-09 | Fairness-Guaranteed Task Assignment for Crowdsourced Mobility Services | Missing from bib | **MISSING** | No DOI in YAML; no confident bib match. |
| CR-10 | Learning to Assign: Towards Fair Task Assignment in Large-Scale Ride Hailing | In bib + linked | `CR-10` | Title + `doi=10.1145/3447548.3467085`. |
| CR-11 | Minimizing Maximum Delay of Task Assignment in Spatial Crowdsourcing | In bib + linked | `CR-11` | Title + `doi=10.1109/ICDE.2019.00131`. |
| CR-12 | Multi-Space Crowd Sensing Task Allocation: A Dynamic Co-Optimization Framework with Fairness-Aware Reinforcement Learning | Missing from bib | **MISSING** | Do **not** confuse with `onGroupFairnessSC` (different title/authors theme: “group fairness”, IP&M 2023). |
| CR-13 | Optimizing Dynamic Task Assignment in Spatial Crowdsourcing: Bilateral Preference-Aware Approaches | **Collision** | **`CR-13` key wrong** | Existing `@article{CR-13,...}` is *“Dynamic task assignment in spatial crowdsourcing”* (Tong & Zhou, SIGSPATIAL 2018) — **not** this bilateral-preference paper. Add a **new** bib entry (new key); consider renaming freeing `CR-13` after migration. |
| CR-14 | Robust Fairness in Spatial-Temporal Resource Allocation (thesis) | Missing from bib | **MISSING** | Thesis-style PDF in library; likely needs `@phdthesis` / `@mastersthesis` once institution metadata is confirmed. |
| CR-15 | Two-Stage Bilateral Online Priority Assignment in Spatio-Temporal Crowdsourcing | In bib + linked (use full entry) | `CR-15` | Prefer the IEEE TSC block with `doi=10.1109/TSC.2022.3197676`; drop duplicate stub entry sharing the same key. |
| CR-16 | Long-Term Fairness in Ride-Hailing Platform | Missing from bib | **MISSING** | No DOI in YAML; no confident bib match. |
| CR-TRIAL | FATP (trial YAML; duplicate of CR-03 theme) | Missing from bib | **MISSING** (same as CR-03) | Encoding artefact — same paper family as **CR-03** / **CR-08** lineage; do not triple-count in bib unless intentional. |

---

## [2] Structural Ancestors

**`notes/*.yml`:** none — metadata lives in `notes/SA-##_*.md` only.

**Loose PDFs (folder root, no `notes/SA-##_*` counterpart for that file):**

- `A General Incentives-Based Framework for Fairness in Multi-agent Resource Allocation.pdf`
- `Dynamic Resource Allocation in Systems-of-Systems Using a Heuristic-Based Interpretable Deep Reinforcement Learning.pdf`
- `Enhancing Game Policy Optimization in Mobile Crowdsourcing- A Reinforcement Learning Approach.pdf`

| ID | Title (from note `.md` H1) | Match status | Bib key or MISSING | Notes |
|----|----------------------------|--------------|-------------------|--------|
| SA-01 | Two-Sided Online Micro-Task Assignment in Spatial Crowdsourcing | Missing from bib | **MISSING** | No YAML DOI; no confident bib title hit. |
| SA-02 | Adaptive Task Assignment in Spatial Crowdsourcing: A Human-in-The-Loop Approach | Missing from bib | **MISSING** | |
| SA-03 | PPO-TA: Adaptive task allocation via Proximal Policy Optimization for spatio-temporal crowdsourcing | Missing from bib | **MISSING** | |
| SA-04 | Two-sided online bipartite matching in spatial data: experiments and analysis | In bib + linked | `SA-04` | Bib key equals ID; `doi=10.1007/s10707-019-00359-w`. |
| SA-05 | Assigning Tasks to Workers based on Historical Data: Online Task Assignment with Two-sided Arrivals | Missing from bib | **MISSING** | |
| SA-06 | Matching Tasks and Workers under Known Arrival Distributions: Online Task Assignment with Two-sided Arrivals | Missing from bib | **MISSING** | |
| SA-07 | Deep Reinforcement Learning for Task Assignment in Spatial Crowdsourcing and Sensing | In bib, stub only | `SA-07` | Entry is placeholder (“Sun et al.”) — treat as **needs real BibTeX** (same key/ID linkage intent is clear). |
| SA-08 | Task Assignment for hybrid scenarios in spatial crowdsourcing: a Q-learning-based approach | In bib, stub only | `SA-08` | Placeholder (“Wang et al.”) — **needs real BibTeX**. |

---

## [3] The Why

**`notes/*.yml`:** none — `notes/TW-##_*.md` only.

**Loose PDFs (root; no `notes/TW-##_*` file for these topics):**

- `A survey on interpretable reinforcement learning.pdf`
- `AdaTaskRec- An Adaptive Task Recommendation Framework in Spatial Crowdsourcing.pdf` (bib entry exists — see below)
- `Reinforcement Learning Incentive Mechanism.pdf`
- `Sustainable Volunteer Engagement- Ensuring Potential Retention and Skill Diversity for Balanced Workforce Composition in Crowdsourcing Paradigm.pdf`

| ID | Title (from note `.md` H1) | Match status | Bib key or MISSING | Notes |
|----|----------------------------|--------------|-------------------|--------|
| TW-01 | On On-line Task Assignment in Spatial Crowdsourcing | Missing from bib | **MISSING** | No matching entry in current `references.bib` (checked by title substring). |
| TW-02 | Real-Time Rideshare Driver Supply Values Using Online Reinforcement Learning | Missing from bib | **MISSING** | |
| TW-03 | Unraveling the Implications of Silent Labor Time in the Gig Economy | In bib, **tentative** | `TW-03` | Bib title string differs (“Unravelling the Implications of Effort Allocation in Gig Economy…”, SSRN `doi=10.2139/ssrn.4679321`, 2024). Likely **same empirical food-delivery / gig work** line — mark **tentative** until PDF front matter or DOI is cross-checked. |

**Related (loose PDF, not TW-ID):** `AdaTaskRec- …` ↔ `@article{AdaTaskRec,...}` (`doi=10.1145/3593582`) — **in bib** but **not tied** to a `TW-##` slug.

---

## [4] OBM Theory

**`notes/*.yml`:** none — `notes/OT-##_*.md` only.

**Loose PDFs (root):**

- `An Axiomatic and Empirical Analysis of Mechanisms for Online Organ Matching.pdf`
- `Meta-Reinforcement_Learning_in_Non-Stationary_and_Dynamic_Environments.pdf`

| ID | Title (from note `.md` H1) | Match status | Bib key or MISSING | Notes |
|----|----------------------------|--------------|-------------------|--------|
| OT-01 | Dynamic Matching in a Two-Sided Market | Missing from bib | **MISSING** | |
| OT-02 | Multi-Objective Online Ride-Matching | Missing from bib | **MISSING** | |
| OT-03 | Delay-Sensitive Task Assignment for Spatial Crowdsourcing | In bib + linked | `OT-03` | Key equals ID; `doi=10.1155/2022/3191761`. |
| OT-04 | Multi-stage online task assignment driven by offline data under spatio-temporal crowdsourcing | Missing from bib | **MISSING** | |
| OT-05 | Multi-Worker-Aware Task Planning in Real-Time Spatial Crowdsourcing | Missing from bib | **MISSING** | |

---

## [5] Technical Reference

**`notes/*.yml`:** none — `notes/TR-##_*.md` only.

**Loose PDFs (root):**

- `Adaptive Regime-Aware Stock Price Prediction Using Autoencoder-Gated Dual Node Transformers with Reinforcement Learning Control.pdf`
- `Agentic Artificial Intelligence for Smart Grids- A Comprehensive Review of Autonomous, Safe, and Explainable Control Frameworks.pdf`
- `MARS- A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management.pdf`
- `Spatial Crowdsourcing Survey.pdf` → see **Tong survey** mapping below.

| ID | Title (from note `.md` H1) | Match status | Bib key or MISSING | Notes |
|----|----------------------------|--------------|-------------------|--------|
| TR-01 | Task Assignment with Worker Churn Prediction in Spatial Crowdsourcing | Missing from bib | **MISSING** | |
| TR-02 | Worker-Churn-Based Task Assignment With Context-LSTM in Spatial Crowdsourcing | Missing from bib | **MISSING** | |
| TR-03 | Spatio-temporal Adaptive Pricing for Balancing Mobility-on-Demand Networks | Missing from bib | **MISSING** | |
| TR-04 | Adaptive Regulation via Dual-Layer Evolution (ARDE): A Multi-Agent Approach to Balancing Efficiency, Fairness, and Diversity in Crowdsourced Platforms | Missing from bib | **MISSING** | |
| TR-05 | Theoretical Foundations of Algorithmic Fairness in Two-Sided Hiring Marketplaces: Interventions for Reducing Discrimination in Job Matching | Missing from bib | **MISSING** | |

---

## [6] Last minute

**`notes/*.yml`:** none — no `LM-##` encoded note files.

**Unencoded PDFs (entire folder):**

| File | Match status | Bib key or MISSING | Notes |
|------|--------------|---------------------|--------|
| `tong2019survey.pdf` | In bib + linked (by DOI/title) | `scSurvey2020` | Bib: *Spatial crowdsourcing: a survey*, VLDB Journal 2020; DOI **`10.1007/s00778-019-00568-7`** (stored with `https://doi.org/` prefix in bib — normalize for comparison). Filename says `2019`; published year in bib is **2020**. Same work as **`[5]`** root `Spatial Crowdsourcing Survey.pdf` — duplicate holdings, no encoded folder ID. |
| `adaTaskRec.pdf` | In bib + linked | `AdaTaskRec` | Same object as **`[3]`** `AdaTaskRec- …pdf` — duplicate PDF without a `TW-##`/`TR-##` code. |

---

## Bib entries present but not mapped to a folder ID

These keys appear in `references.bib` but were **not** matched to any `CR-##` / `SA-##` / `TW-##` / `OT-##` / `TR-##` slug in folders [1]–[5] on the basis of DOI/title:

- `geo-crowdKazemi` — GeoCrowd / SIGSPATIAL 2012 (classic SC reference).
- `li2019multi` — mobile crowd sensing multi-task allocation (Procedia / conference).
- `cheng2019cooperation` — ICDE cooperation-aware SC task assignment.
- `deng2009fairness`, `JFI-Original`, `mo2000fair` — fairness background.
- `onGroupFairnessSC` — group-fair SC task assignment (IP&M 2023).
- `tong2020` — placeholder stub (should be reconciled with `scSurvey2020`).
- `didi2016gaia` — dataset note.

---

## Summary

| Metric | Count |
|--------|------:|
| **`notes/*.yml` paper files** (folder [1] only; `index.yml` excluded) | **17** |
| **Encoded `notes/*.md` rows** (folders [2]–[5] only) | **21** |
| **Encoded ID rows total** (`CR-01`…`CR-16` + `CR-TRIAL` + `SA-`…+`TR-`) | **38** (`CR-TRIAL` duplicates the FATP / `CR-03` theme) |
| **Clear in-bib match** (folder ID equals bib key, or confident DOI/title for that slug) | **9** — `CR-04`, `CR-05`, `CR-08`, `CR-10`, `CR-11`, `CR-15`, `SA-04`, `OT-03`, plus **`CR-06`** ↔ `Game-FairnessAware-2021` (same paper; bib key ≠ encoded ID) |
| **Bib exists but stub / tentative** | **3** — `SA-07`, `SA-08` placeholders; **`TW-03`** (note title vs bib title differ — tentative until PDF/DOI checked) |
| **Wrong or missing bib for encoded slug** | **26** — remaining encoded rows (includes **`CR-13`** key collision with a different Tong/Zhou article; **`CR-03`** / **`CR-TRIAL`** vs **`CR-08`** as separate venues unless merged) |
| **Loose root PDFs** (`[2]`–`[5]`, no `notes/XX-##_*` for that PDF) | **13** |
| **Unencoded PDFs** (`[6]` only; no `LM-##` scheme) | **2** |

**Additional holdings in bib** (no `[1]`–`[5]` slug in this audit): loose **`AdaTaskRec`** PDFs in `[3]` / `[6]` map to `@article{AdaTaskRec,...}`; **`Spatial Crowdsourcing Survey.pdf`** (`[5]`) matches `scSurvey2020` like **`tong2019survey.pdf`** (`[6]`).

**Tong survey example:** `[6]/tong2019survey.pdf` and `[5]/Spatial Crowdsourcing Survey.pdf` both align with **`scSurvey2020`** (*Spatial crowdsourcing: a survey*, DOI `10.1007/s00778-019-00568-7`), despite filename year **2019** vs bib year **2020**.

---

*Generated from filesystem + `references.bib` content available at audit time; no external lookups.*
