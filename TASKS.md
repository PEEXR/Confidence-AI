# TASKS.md — work distribution for the confidence project

All tasks from PLAN.md v3, split across three people for a one-week sprint.
Every task maps to the pre-registered experiment/gate it serves (see PLAN
§13–§17). Estimator and pass rules are fixed before any data — see PLAN §16
before trusting any number. The grid is 5 model variants × 6 tiers = 30 cells;
cells are committed as their 100-question pilot clears the 25–80% accuracy
band and the compute ledger (X2) allows — a ragged grid is the planned
outcome, not a failure (PLAN §3, §10).

## 1. All tasks (by phase)

| Task | What it involves | Produces | Depends on | Effort |
|---|---|---|---|---|
| **A1** | Kaggle env: 2× T4 config, `/kaggle/working` persistence, device-detection (T4/P100 fallback), checkpoint/resume scaffold. **Create the shared Kaggle org/team** (not present yet) + shared Dataset namespace | Working env, org, mount point | — | M |
| **A2** | Download Qwen2.5 ladder (0.5B/1.5B/3B/7B-Instruct + 7B-base); verify inference + hook-based 5-percentile extraction; upload weights as shared Kaggle Datasets (read-only mount) | Model-load script, mount docs, finiteness baseline | A1 | L |
| **A3** | Acquire six-tier ladder (PopQA, SimpleQA, GSM8K, MATH); run 100-q pilots per cell, keep cells in 25–80% accuracy band; build per-cell 60/20/20 splits with tier/difficulty tags | `questions.jsonl` per cell + pilot verdicts | A1, A2 | L |
| **B1** | Build grader for all four answer forms: entity/short (PopQA/SimpleQA), numeric (GSM8K), LaTeX (MATH); normalized string-match → NLI entailment → LLM-judge escalation (PLAN §7) | `grade.py` | A3 | L |
| **B2** | Manually check 50–100 questions per grader family; compare to grader; run **Gate 1** (≥95% agreement) | Agreement report + gate verdict | B1, A3 | M |
| **C1** | Write format A/B/C prompt templates + a shared runner for all 5 model variants, incl. base-elicitation variant (PLAN §4, §9) | `verbal.py` | A2 | S |
| **C2** | Run the 3 formats on the 50–100-question agreement subset (temp=0, single sample) on an Instruct model + the base model (E1/E5) | Raw verbalized scores | C1, A3 | S |
| **C3** | Empirical bucket→probability mapping for format B (PLAN §4) | Bucket table | C2 | S |
| **C4** | Pairwise Spearman across formats; run **Gate 2** (≥0.6); pick canonical format; record the base-elicitation verdict (E5) | H0 verdict + E5 verdict | C2, C3 | M |
| **D1** | N=10 sampling runner at T=0.7–1.0, batched, checkpoint/resume by question ID, spanning both T4s (PLAN §5, §10) | `behavior.py` + raw samples | A2, A3 | L |
| **D2** | Semantic entropy: cluster answers (string → NLI), compute Shannon entropy (PLAN §5) | Behavioral confidence scores | D1 | M |
| **D3** | Sanity checks: all-same→H=0, even-split→H≈max (PLAN §5) | Sanity report | D2 | S |
| **E1** | Forward-hook extraction at the 5 percentile-layer vectors, last prompt token, single greedy pass; log non-finite fraction per layer (PLAN §6, §10) | `probe/extract.py` + finiteness log | A2, A3 | L |
| **E2** | Train logistic probe per (cell × percentile) on the **calibration** split only (PLAN §6) | Probe AUROC per (cell × percentile) | E1 | L |
| **E3** | Probe validity: finiteness pre-check, label-shuffle null, surface/embedding baseline; run **Gate 3** (≥1 percentile AUROC ≥ 0.65) | Gate 3 verdict | E2 | M |
| **E4** | Isotonic-calibrate winning probe; apply to test split (PLAN §6) | Internal confidence scores | E3 | M |
| **F1** | Grade all answers with the automated grader across all committed cells (PLAN §7) | Ground-truth labels | B1, A3 | L |
| **F2** | Fit per-signal calibration functions (isotonic/Platt) on calibration split (PLAN §8) | Calibration functions | F1, C2, D2, E4 | M |
| **F3** | Murphy decomposition of Brier (reliability/resolution/uncertainty) per signal per cell (PLAN §8) | H1 stats | F2 | M |
| **F4** | Pooled hierarchical logistic regression across the grid, question-level random effects (PLAN §8) | Grid model + interaction terms (H4) | F2 | L |
| **F5** | Pairwise Spearman (raw) + Pearson (calibrated) correlations (PLAN §8) | Correlation table | F2 | S |
| **F6** | Omniscience-Index (PLAN §8.2) | Index values | F1 | S |
| **F7** | Quadrant analysis: bucketing + pulling example questions (PLAN §8) | H2 material | F2, F1 | M |
| **F8** | Depth curves: AUROC vs layer percentile, per tier, faceted by model (PLAN §6, §15 Fig 4) | H4 figure data | E2 | M |
| **G1** | Forced-answer pass on every Format C Pass across committed cells (PLAN §4.1) | Forced answers | C2 | M |
| **G2** | Justified-hedge vs missed-knowledge split (PLAN §4.1) | Abstention split table | G1, F1 | M |
| **H1** | Base-model elicitation check (E5): Qwen2.5-7B-base usable output for formats A/B/C on a small subset (PLAN §9, §16) | E5 verdict → Gate 4 | A2 | M |
| **H2** | Run D/E/F/G pipeline on 7B-base for committed cells (PLAN §9) | 7B-base signal tables | H1 | L |
| **H3** | Delta analysis: hopeful-confidence rate 7B-base vs 7B-Instruct; missed-knowledge guard; run **Gate 4** (PLAN §13 H3, §16) | H3 verdict | H2, F2 | M |
| **I1** | Write `plots/fig4_depth_prediction.py` stub + refresh Fig 3 stub for base-vs-Instruct (PLAN §15, §17.3) | Pre-registered stubs | — | S |
| **I2** | Re-run the four `plots/` stub scripts on real data (PLAN §15) | Figures 1–4 | F3, F7, F8, H3 | S |
| **I3** | Fill §0 status + §17.2 run log in PLAN.md (PLAN §17) | Updated pre-registration | everything | S |
| **I4** | Write the workshop paper (PLAN §12) | Paper draft | I2, I3 | L |
| **X1** | Provenance tracking throughout (seed, config hash, model ver, code SHA) | Reproducible results | all | ongoing |
| **X2** | Compute-budget ledger: track GPU-hr/wk vs the 30-hr cap; decide cell-commitment priority; enforce checkpoint/resume (PLAN §10) | Cell-commitment order + budget report | all | ongoing |

## 2. The 3-way split (by topic)

| Person | Owns | Gates | Hypotheses |
|---|---|---|---|
| **Person A — Data & verbalized** | A1, A2, A3, B1, B2, C1, C2, C3, C4, F1, X2 | Gate 1, Gate 2 | H0 |
| **Person B — Behavior, probe & stats** | D1, D2, D3, E1, E2, E3, E4, F2, F3, F4, F8, X2 | Gate 3 | H1, H2, H4 |
| **Person C — Abstention, comparison, figures & paper** | G1, G2, H1, H2, H3, F5, F6, F7, I1, I2, I3, I4 | Gate 4 | H2, H3 |

**Shared responsibilities:**
- **D2/D3 (semantic entropy):** B authors the entropy code once; whoever owns
  a cell's sampling runs it on their own cells (C on 7B-base, B on the ladder).
- **X2 (compute ledger):** B leads the ledger (burns the most GPU via D/E); A
  and C report GPU use and pilot results so the ledger decides cell commitment.
- **Model/dataset access:** A downloads and uploads weights + data as shared
  Kaggle Datasets once; B and C mount them read-only — they do **not**
  download (PLAN §10).

## 3. One-week schedule

Cells commit in **waves** as pilots clear the 25–80% band and the X2 ledger
allows. 7B jobs (ladder top rung + base) are the long poles — queue them
early. Slot by GPU time, not by cell count.

| Day | Person A | Person B | Person C |
|---|---|---|---|
| **1** | **All together:** A1 env + **Kaggle org/team creation** + `/kaggle/working` + device detection; A2 start downloads/uploads; A3 start dataset acquisition | A2 (parallel, verify load/devices); A3 pilot runs | A2 (parallel); I1 commit Fig 4 + Fig 3 stubs (pre-registration) |
| **2** | A3 finish pilots → **first cell commitments**; B2 manual check → **Gate 1**; start B1 grader | D1 sampler (batched, resumable); E1 extraction scaffold + finiteness logging | H1 base-elicitation (riskiest, start early); E1 on 7B; queue 7B jobs on X2 ledger |
| **3** | B1 finish → C1 prompts ×5 → C2 subset runs (Instruct + base) | D2/D3 entropy + sanity; E2 probe training; stub F4/F8 scripts (off critical path) | H2 pipeline prep; run C/D on committed 7B-base cells |
| **4** | C3 bucket mapping → C4 Spearman → **Gate 2** (H0 secured); start F1 grading | E3 finiteness pre-check → probe validity → **Gate 3** | G1 forced-answer pass (needs A's C2 Format C passes) |
| **5** | F1 continue grading | E4 isotonic; F3 Murphy decomposition; F4 hierarchical regression | H2 run D/E/F/G on 7B-base |
| **6** | F1 finish; X2 budget reconciliation | F8 depth curves; I2 re-run plot stubs on first real data | F5/F6/F7 correlations, Omniscience-Index, quadrant analysis; H2 finish → feed H3 |
| **7** | **All together:** H3 delta + missed-knowledge guard → **Gate 4**; I3 (fill §0 + run log); I4 (write paper — A writes data/method, B writes stats/probe, C writes model-comparison) | | |

## 4. Rules for the sprint

- **Protect the abort branch.** Gate 2 / H0 (format agreement) is the
  publishable-alone deliverable — never let it slip. If behind, drop in this
  order: **H3 first → H2 → H1.**
- **Start the two hardest tasks on Day 2** (H1 base-elicitation, E1 on 7B) —
  they are also the riskiest, and novices need the extra days.
- **Cell commitment is gated.** A cell is added only after its 100-q pilot
  lands in the 25–80% accuracy band **and** the X2 ledger has GPU budget. The
  grid is expected to be ragged — that is the planned outcome (PLAN §3, §10).
- **Handoff to watch:** C's G1 needs A's C2 output (Format C passes) —
  schedule the exchange for Day 4 morning.
- **Everyone owns a committed `run.py` before any number** for their
  experiments, and pre-commits expected results (PLAN §17.3). Ad-hoc runs get
  retracted.
- **Every long Kaggle job checkpoints and resumes by question ID**; a job that
  cannot resume is not a measurement (PLAN §10, §17.3).
- **Run the Gate 3 finiteness pre-check before trusting any probe AUROC** —
  dirty activations and "no signal" must be distinguishable (PLAN §16).
- **Measure the noise floor before any Δ.** A difference without a denominator
  is a number, not a measurement (PLAN §17.3).
- **Tune on held-out seeds only; never let the trained system choose the
  evaluation set.** Layer/percentile selection and calibration happen on the
  calibration split only (PLAN §14.2).
