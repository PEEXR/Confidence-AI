# TASKS.md — work distribution for the confidence project

All tasks from PLAN.md, split across three people for a one-week sprint. Every
task maps to the pre-registered experiment/gate it serves (see PLAN §13–§17).
Estimator and pass rules are fixed before any data — see PLAN §16 before
trusting any number.

## 1. All tasks (by phase)

| Task | What it involves | Produces | Depends on | Effort |
|---|---|---|---|---|
| **A1** | Set up Python env, GPU, packages (torch, transformers, scipy, sklearn, matplotlib) | Working env + pinned `requirements` | — | M |
| **A2** | Download Qwen3.5-0.8B (+ Llama alt) and LFM2.5-8B-A1B; verify inference + hidden-state extraction on an 8GB card | Model-load script, VRAM headroom report | A1 | M |
| **A3** | Acquire PopQA / SimpleQA / AA-Omniscience; check AA-Omniscience HF-subset coverage (PLAN §3) | Dataset files + coverage note | — | M |
| **A4** | Select + stratify 300–500 questions; build 60/20/20 split (train/cal/test) | `questions.jsonl` with split + domain/difficulty tags | A3 | M |
| **B1** | Build grader: normalized string match → NLI entailment → LLM-judge escalation (PLAN §3) | `grade.py` | A4 | L |
| **B2** | Manually check 50–100 questions; compare to grader; run **Gate 1** (≥95% agreement) | Agreement report + gate verdict | B1, A4 | M |
| **C1** | Write format A/B/C prompt templates + a shared runner (PLAN §4) | `verbal.py` | A2 | S |
| **C2** | Run the 3 formats on the 50–100 question subset, temp=0, single sample (PLAN §4) | Raw verbalized scores | C1, A4 | S |
| **C3** | Empirical bucket→probability mapping for format B (PLAN §4) | Bucket table | C2 | S |
| **C4** | Pairwise Spearman across formats; run **Gate 2** (≥0.6); pick canonical format | H0 verdict | C2, C3 | S |
| **D1** | N=10 sampling runner at T=0.7–1.0 (PLAN §5) | `behavior.py` + raw samples | A2, A4 | M |
| **D2** | Semantic entropy: cluster answers (string → NLI), compute Shannon entropy (PLAN §5) | Behavioral confidence scores | D1 | M |
| **D3** | Sanity checks: all-same→H=0, even-split→H≈max (PLAN §5) | Sanity report | D2 | S |
| **E1** | Hidden-state extraction at last prompt token, all layers, single greedy pass (PLAN §6) | `probe/extract.py` | A2, A4 | M |
| **E2** | Layer sweep: train logistic probe per layer on **calibration** split only (PLAN §6) | Probe AUROC per layer | E1 | L |
| **E3** | Probe validity: label-shuffle null + surface/embedding baseline + train AUROC≥0.65; run **Gate 3** | Gate 3 verdict | E2 | M |
| **E4** | Isotonic-calibrate winning probe; apply to test split (PLAN §6) | Internal confidence scores | E3 | M |
| **F1** | Grade all answers with the automated grader (PLAN §7) | Ground-truth labels | B1, A4 | M |
| **F2** | Fit per-signal calibration functions (isotonic/Platt) on calibration split (PLAN §8) | Calibration functions | F1, C2, D2, E4 | M |
| **F3** | ECE + Brier per signal on test split with bootstrap CIs (PLAN §8) | H1 stats | F2 | M |
| **F4** | Pairwise Spearman (raw) + Pearson (calibrated) correlations (PLAN §8) | Correlation table | F2 | S |
| **F5** | Omniscience-Index (PLAN §8.2) | Index values | F1 | S |
| **F6** | Quadrant analysis: bucketing + pulling example questions (PLAN §8) | H2 material | F2, F1 | M |
| **G1** | Forced-answer pass on every Format C Pass (PLAN §4.1) | Forced answers | C2 | M |
| **G2** | Justified-hedge vs missed-knowledge split (PLAN §4.1) | Abstention split table | G1, F1 | M |
| **H1** | Verify hidden-state extraction through LFM MoE routing + fresh layer sweep (PLAN §9) | MoE feasibility report | A2 | M |
| **H2** | Repeat tasks C/D/E/F/G on the LFM model | LFM signal tables | H1 | L |
| **H3** | Delta analysis: hopeful-confidence rate baseline vs LFM; missed-knowledge guard; run **Gate 4** | H3 verdict | H2, F2 | M |
| **I1** | Re-run the three `plots/` stub scripts on real data (PLAN §15) | Figures 1–3 | F3, F6, H3 | S |
| **I2** | Final figures: calibration curves, quadrant plot, model delta, abstention split | Publication figures | I1 | M |
| **I3** | Fill §0 status + §17.2 run log in PLAN.md | Updated pre-registration | everything | S |
| **I4** | Write the workshop paper (PLAN §12) | Paper draft | I2, I3 | L |
| **X1** | Provenance tracking throughout (seed, config hash, model ver, code SHA) | Reproducible results | all | ongoing |

## 2. The 3-way split (by topic)

| Person | Owns | Gates | Hypotheses |
|---|---|---|---|
| **Person A — Data & verbalized** | A1, A3, A4, B1, B2, C1, C2, C3, C4 | Gate 1, Gate 2 | H0 |
| **Person B — Behavior, probe & statistics** | D1, D2, D3, E1, E2, E3, E4, F1, F2, F3, F4, F5, F6 | Gate 3 | H1, H2 |
| **Person C — Abstention, LFM & paper** | G1, G2, H1, H2, H3, I1, I2, I3, I4 | Gate 4 | H2, H3 |

## 3. One-week schedule

| Day | Person A | Person B | Person C |
|---|---|---|---|
| **1** | **All together:** A1 env setup + skim GUIDE/PLAN + 2-hour Python/transformers tutorial | A3 (datasets) + A2 (start model downloads) | A2 (same downloads, parallel) |
| **2** | A4 (select/stratify 300–500 q, build split) → start B1 | D1 (N=10 sampler) + E1 (hidden-state extraction scaffold) | H1 (LFM MoE hidden-state check — riskiest task, start early) |
| **3** | Finish B1; B2 manual check of 50–100 → **Gate 1**; start C1 (prompts) | D2 + D3 (semantic entropy + sanity) | Finish H1; prepare H2 pipeline |
| **4** | C2 (run 3 formats on subset) → C3 (bucket mapping) → C4 Spearman → **Gate 2** (H0 secured) | E2 layer sweep → E3 probe validity → **Gate 3** | G1 (forced-answer pass, once A hands over Format C passes) |
| **5** | F1 (grade all answers) | E4 isotonic; F2 calibration funcs; F3 ECE/Brier; F4–F5 correlations + Omniscience-Index | H2 (run C/D/E/G pipeline on LFM) |
| **6** | X1 provenance cleanup; help anywhere | F6 quadrant analysis; I1 (re-run plot stubs on real data) | H3 delta + missed-knowledge guard → **Gate 4**; I2 figures |
| **7** | **All together:** I3 (fill §0 + run log), I4 (write paper — A writes data/method, B writes stats, C writes model-comparison) | | |

## 4. Rules for the sprint

- **Protect the abort branch.** Gate 2 / H0 (format agreement) is the
  publishable-alone deliverable — never let it slip. If behind, drop in this
  order: **H3 first → H2 → H1.**
- **Start the two hardest tasks on Day 2** (E2 layer sweep, H1 MoE check) —
  they are also the riskiest, and novices need the extra days.
- **Handoff to watch:** C's G1 needs A's C2 output (Format C passes) —
  schedule the exchange for Day 4 morning.
- **Everyone owns a committed `run.py` before any number** for their
  experiments, and pre-commits expected results (PLAN §17.3). Ad-hoc runs get
  retracted.
- **Measure the noise floor before any Δ.** A difference without a denominator
  is a number, not a measurement (PLAN §17.3).
- **Tune on held-out seeds only; never let the trained system choose the
  evaluation set.** Layer selection and calibration happen on the calibration
  split only (PLAN §14.2).
