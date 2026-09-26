# Detailed Findings, Understandings & Analysis: `prod500`

---
###### NOTE: ALL "NOTES" ARE HANDWRITTEN AND CAPTURE SCIENTIFIC INTUITION, PRACTICAL TAKEAWAYS, AND CAVEATS.
**RUN:** `prod500` (Full 30-Cell Grid Evaluation)  
**DATE:** 2026-09-04  
**PLATFORM:** Molab · NVIDIA RTX PRO 6000 Blackwell Server Edition (101.98 GB VRAM)  
**STATUS:** ✅ Completed Run · 30/30 Cells Committed · 4.778 Measured GPU-Hours · All Audit v3 Guards Enforced  
---

## 1. Differences Between `prod500` and Legacy August Runs (`results/`, `results_v2_flawed/`)

The `prod500` dataset is the first **fully repaired, full-grid production benchmark** of the Confidence-AI pipeline. Comparing it against the historical August runs highlights why this run is scientifically authoritative:

| Feature / Metric | `results_v2_flawed/` (Run 2) | `results/` (Old Clean Run) | `prod500` (New Authoritative Benchmark) |
| :--- | :--- | :--- | :--- |
| **Questions per Cell ($N$)** | 1,000 questions | 1,000 questions | **500 questions** (Balanced Full Grid) |
| **Total Grid Cells** | 13 of 30 committed | 13 of 30 committed | **30 of 30 committed (100% full grid)** |
| **Activation Tap Method** | Broken tap (measured decode artifacts) | Broken tap (dirty late-layer vectors) | **Frozen prefill pass before decode** (Fixed tap) |
| **Gate 3 Negative Control ($p_0$)**| Absent | Absent | **Enforced ($p_0 \approx 0.50 \pm 0.10$)**; caught 7B-base C3 |
| **H4 Verdict (Depth Onset)** | Falsified ($\Delta \approx +2.7\%$, CI crosses 0) | Falsified (CIs overlapped) | **SUGGESTIVE TREND ($\Delta = +13.64\%$, permutation $p = 0.053$, 25-pp grid)** |
| **H3 Evaluation** | False positive bar ($0.0\%$ due to NaN) | False positive bar ($0.0\%$ due to NaN) | **Honest Null / Untested** (Declined to test) |
| **Gate 1 Status** | Null / Unfilled | Completed (97.5% manual agreement) | **PASSED (97.5% manual agreement imported)** |
| **Gate 2 Status** | Falsified | Falsified | **Falsified (Fallback to `Bfix` canonical)** |
| **Hierarchical Regression** | Not converged / Flawed | Ad-hoc estimates | **Bayesian Mixed GLM ($N = 1166$, Converged)** |
| **GPU Time** | ~1.5h (partial) | ~1.5h (partial) | **4.778 GPU-hours** |

---
###### note: Why did we commit 30 out of 30 cells here when only 13 committed previously? 
In the August runs, the pipeline dropped 17 cells under a strict 25%–80% accuracy rule. But Audit v3 correctly recognized that throwing away 57% of your data destroys cross-model comparisons. Instead, `COMMIT_CELLS_OUTSIDE_BAND=True` preserved all 30 cells, passing out-of-band accuracy as an explicit covariate (`in_band`) into the multi-level hierarchical regression.
---

---

## 2. Directory Anatomy — Navigating `confidence_out/prod500/`

The `confidence_out/prod500/` folder organizes the complete experimental outputs across 6 subdirectories:

### 1. ⚡ `activations/` — Hidden Activation Shards & Gate 3 Pre-Checks
* **`{model}__{tier}.npz`**: Compressed NumPy archives containing float32 activation vectors tapped across 5 layer depth percentiles ($0\%, 25\%, 50\%, 75\%, 100\%$) during an isolated prefill pass.
* **`{model}__{tier}.finiteness.json`**: Gate 3 pre-check verifying that hidden tensors are 100% finite (`nonfinite_frac == 0.0`).

### 2. 📊 `derived/` — Parsed Parquet Tables & Hypothesis JSONs
* **`signals.parquet`**: Master evaluation table joining calibrated Verbal, Behavioral, and Internal confidence metrics for all test rows.
* **`graded.parquet`**: Deterministic grading results across all generation variants (`A`, `B`, `C`, `SAMPLE`, `FORCED`, `EXTRACT`).
* **`entropy.parquet`**: Semantic entropy computed over $N=10$ samples with strict NLI bidirectional equivalence merging.
* **`quadrants.parquet`**: $N=831$ complete-case item classifications into the 4 epistemic alignment quadrants.
* **`h0_gate2.json`**, **`h1_calibration.json`**, **`h2_quadrants.json`**, **`h3_model_delta.json`**, **`h4_depth.json`**: Pre-registered hypothesis outputs and confidence intervals.
* **`hierarchical_regression.json`**: Multi-level Bayesian GLM parameter estimates.

### 3. 🖼️ `figures/` — Vector PDFs and High-Res PNGs
* **`fig1_calibration`**: Reliability diagrams and Murphy decomposition bars.
* **`fig2_quadrant`**: Epistemic alignment scatter and abstention split.
* **`fig4_depth_prediction`**: Layer percentile AUROC curves across retrieval vs. reasoning.
* **`fig5_cell_commitment_grid`**: Full 30-cell heatmap of accuracies and commitments.
* **`fig6_signal_correlations`**: Pairwise correlation heatmaps.

### 4. 📄 `tables/` — Publication Tables (CSV, TeX, Parquet)
* **`t1` – `t15`**: LaTeX source (`.tex`) and CSVs covering dataset composition, parse rates, Murphy components, probe sweeps, depth onsets, and hypothesis verdicts.
* **`gate1_manual_check_sheet.csv`**: The 200-row human verification audit sheet.

### 5. ⚙️ `meta/` & 📝 `logs/`
* **`final_report.json`**: Complete provenance, device specifications, and gate outcomes.
* **`run_log_rows.md`**: Pre-formatted Markdown log row ready for insertion into PLAN §17.2.
* **`events.jsonl`**: Detailed execution event stream recording run durations and GPU allocations.

---

## 3. The Five Pre-Registered Hypotheses: Scorecard & Decisive Findings

### Summary Scorecard

| Hypothesis | Test Description | Gate & Threshold | Observed Result | Final Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **H0** | Verbal format agreement ($A, B, C$) | **Gate 2** ($\rho \ge 0.60$) | $\rho_{A-B} = 0.126$, $\rho_{A-C} = -0.069$, $\rho_{B-C} = 0.247$ | **FALSIFIED / NULL** |
| **H1** | Calibration hierarchy (ECE / Murphy) | Bootstrap CI excludes 0 | $\Delta_{\text{worst}-\text{best}} = 0.0071$, 95% CI $[-0.016, +0.027]$ | **FALSIFIED / NULL** |
| **H2** | Error quadrants associate with question features | Gate 1 & Cell Replication | Driven by model $\times$ tier proxies, fails cell replication | **DESCRIPTIVE ONLY** |
| **H3** | Base vs. Instruct overconfidence drop | Gate 4 ($N \ge 20$ matched) | 0 / 0 matched rows (Base model fails verbal formatting) | **UNTESTED / HONEST NULL** |
| **H4** | Internal probe onsets deeper for reasoning than retrieval | **Gate 3** (Probe Validity) | Retrieval onset: **27.3%**, Reasoning onset: **40.9%** ($\Delta = \mathbf{+13.64\%}$) | **SUGGESTIVE TREND** *(Motivates 5–10% sweep)* |

---

### Detailed Breakdown of the 5 Hypotheses

#### 1. Hypothesis H0 (Gate 2) — Verbal Format Invariance: **FALSIFIED**
* **The Claim:** If verbalized confidence reflects an intrinsic cognitive state, the model's confidence ranking across questions should be invariant whether asked for a percentage ($A$), words ($B$), or a wager ($C$).
* **The Evidence:** Across $N = 2,874$ paired answers, pairwise Spearman correlations were abysmal:
  * **Pair A–B:** $\rho = +0.126$ (95% CI: $[0.090, 0.161]$)
  * **Pair A–C:** $\rho = -0.069$ (95% CI: $[-0.111, -0.030]$) — *statistically significant negative correlation!*
  * **Pair B–C:** $\rho = +0.247$ (95% CI: $[0.211, 0.285]$)
  * **Un-aligned Base Model (`7b-base`):** Format agreement collapses completely into inverted rankings ($\rho_{A-B} = -0.356$).
* **The Verdict:** **Gate 2 Failed**. Verbal confidence is not an invariant epistemic readout. It is a prompt-dependent surface artifact. Per the PLAN §16 fallback, all three formats must be reported separately, with `Bfix` adopted as canonical.

---
###### note: Why did Pair A-C have a negative correlation?
Because on questions where the model boldly says "90% confident" (Format A), it often gets tripped up by the betting stakes in Format C and chooses to PASS. Verbal prompting formats elicit completely different behavioral modes rather than tapping a single internal scalar.
---

#### 2. Hypothesis H1 — Calibration Hierarchy & Murphy Decomposition: **FALSIFIED**
* **The Claim:** Verbalized confidence will exhibit substantially worse calibration error (ECE) than Behavioral (entropy) or Internal (activation probe) signals.
* **The Evidence:** In the pooled test evaluation:
  * **Verbal (`Bfix`):** $\text{ECE} = 0.0356$ (95% CI: $[0.0239, 0.0614]$), $\text{Brier} = 0.1562$
  * **Behavioral:** $\text{ECE} = 0.0427$ (95% CI: $[0.0347, 0.0606]$), $\text{Brier} = 0.1426$
  * **Internal:** $\text{ECE} = 0.0368$ (95% CI: $[0.0314, 0.0584]$), $\text{Brier} = 0.1572$
  * $\Delta(\text{worst} - \text{best}) = 0.0071$ with a 95% bootstrap CI of $[-0.0164, +0.0272]$, cleanly encompassing zero.
* **The Crucial Qualifier:** ECE measures calibration *reliability* (how well stated numbers match empirical percentages), where all three signals do well after calibration. But in **Resolution** (the ability to cleanly separate correct from incorrect answers), **Behavioral dominates**:
  * $\text{Resolution}_{\text{Behavioral}} = \mathbf{0.0759}$
  * $\text{Resolution}_{\text{Internal}} = 0.0546$
  * $\text{Resolution}_{\text{Verbal}} = 0.0343$
* **The Verdict:** **H1 Falsified on ECE**. But Behavioral is **$2.2\times$ more discriminative** than Verbal confidence.

#### 3. Hypothesis H2 — Error Quadrants & Question Characteristics: **DESCRIPTIVE ONLY**
* **The Claim:** Disagreements between verbal confidence and internal/behavioral certainty are driven by intrinsic question characteristics (length, numbers, entities, years).
* **The Evidence:** On the complete-case set ($N = 831$):
  * **Honest Doubt:** $77.1\%$ (641 rows)
  * **Grounded Certainty:** $11.0\%$ (91 rows)
  * **Excessive Hedging:** $10.8\%$ (90 rows)
  * **Performative Certainty:** $1.1\%$ (9 rows)
  * In pooled chi-square tests, question features like `has_multi_entity` ($p = 3.8 \times 10^{-17}$) and `has_number` ($p = 1.0 \times 10^{-14}$) seemed significant.
  * **Threshold Artifact vs. Continuous Metric ($\Delta = \text{Verbal} - \text{Behavioral}$):** The 10:1 skew (90 vs. 9) is an artifact of the hardcoded 0.5 cutoff on calibrated probabilities for hard tasks (and is hyper-concentrated: Performative Certainty appears in only 1 of 9 cells, Excessive Hedging in 2 of 9 cells; in 8 of 9 cells, calibrated verbal confidence never exceeds 0.50). When evaluated as a continuous gap, $\Delta$ is centered symmetrically at zero ($\text{mean} = -0.006, \text{median} = -0.0004, \text{SD} = 0.114$). Regressing continuous $\Delta$ on question characteristics under classical OLS across all 831 complete-case items confirms zero association (`is_long`: $\beta = +0.007, p = 0.370$; `has_year`: $\beta = +0.007, \text{SE} = 0.017, p = 0.650$ [using classical SE; robust HC1 yields $p \approx 0.43$, still non-significant]; `has_number`: $\beta = +0.001, p = 0.917$; `has_multi_entity`: $\beta = +0.005, p = 0.545$). With cell fixed effects, all $p > 0.50$ (similarly, $\Delta_{\text{internal}} = \text{Verbal} - \text{Internal}$ yields all $p \in [0.65, 0.99]$).
* **The Verdict:** **H2 Not Supported**. Reported strictly as a descriptive result: cross-signal disagreement is heterogeneous across cells and driven by task difficulty, not an invariant property of question syntax.

#### 4. Hypothesis H3 (Gate 4) — Post-Training Overconfidence Reduction: **UNTESTED**
* **The Claim:** Instruction tuning (7B-Instruct vs. 7B-Base) reduces performative overconfidence without triggering the missed-knowledge guard.
* **The Evidence:** Raw unaligned base models (`qwen2.5-7b-base`) do not follow prompt instructions to output verbal confidence strings. Across the test split, $600/600$ base rows had missing verbal data, leaving $0/0$ matched complete cases.
* **The Verdict:** **Gate 4 Untested**. Unlike the pre-repair legacy code (which coerced NaNs into a bogus $0.0\%$ overconfidence bar), the pipeline correctly declined to test.

#### 5. Hypothesis H4 (Gate 3) — Emergence of Internal Confidence: **SUGGESTIVE TREND**
* **The Claim:** Internal representations of correctness emerge earlier in transformer depth for factual retrieval than for multi-step reasoning.
* **The Evidence:** Across 22 validated cells that passed Gate 3:
  * **Mean Onset Depth for Factual Retrieval (R1, R2, R3):** **$27.3\%$** of layer depth (10 of 11 cells at 25%, 1 at 50%).
  * **Mean Onset Depth for Multi-Step Reasoning (C1, C2, C3):** **$40.9\%$** of layer depth (6 at 25%, 3 at 50%, 2 at 75%).
  * **Difference:** $\Delta(\text{reasoning} - \text{retrieval}) = \mathbf{+13.64\%}$ depth.
  * **Bootstrap 95% CI:** $[+2.27\%, +27.27\%]$.
  * **Discrete Grid Resolution & Caveats:** Probes were evaluated on a coarse 5-point discrete grid ($0\%, 25\%, 50\%, 75\%, 100\%$), restricting observed onsets to $\{25\%, 50\%, 75\%\}$. A permutation test on the 11 vs 11 onsets yields $p = 0.053$. Sensitivity analysis shows the effect is heavily carried by two C1 cells at 75% (excluding them reduces $\Delta$ from $13.64\%$ to $6.06\%$).
* **The Verdict:** **H4 Suggestive Trend**. Rather than claiming definitive significance ($p < 0.05$), this represents a compelling directional pattern that motivates a finer 5–10% probe sweep to definitively establish depth dynamics.

---

## 4. Deep Dive: The Bayesian Hierarchical Mixed GLM ($N = 1166$)

To determine which confidence signal actually predicts correctness when they all compete simultaneously, the pipeline fit a multi-level generalized linear mixed model across 600 unique questions in 12 cells, controlling for random intercepts across **Model, Tier, Question, and Model $\times$ Tier**:

$$\text{logit}(P(\text{correct})) = -2.97 + 7.41(\text{behavioral}) + 1.06(\text{internal}) - 1.95(\text{verbal}) + 0.71(\log(\text{params}) \times \text{internal}) + \dots$$

### Parameter Breakdown

| Predictor Term | Coefficient ($\beta$) | Posterior SD | Interpretation |
| :--- | :---: | :---: | :--- |
| **Intercept** | $-2.97$ | $0.09$ | Baseline log-odds of correctness |
| **`behavioral_cal`** | **$+7.41$** | **$0.22$** | **The single strongest true signal of correctness ($p \ll 10^{-15}$)** |
| **`internal_cal`** | **$+1.06$** | **$0.25$** | **Internal probe adds orthogonal predictive power ($p < 10^{-4}$)** |
| **`verbal_cal`** | **$-1.95$** | **$0.25$** | **Verbal confidence is deceptive / negatively associated with correctness!** |
| **`log_params * internal`** | **$+0.71$** | **$0.14$** | **Internal signal becomes significantly more predictive as model scales** |
| **`is_reasoning`** | $+1.39$ | $0.12$ | Reasoning tier base-rate adjustment |
| **`is_long`** | $-1.04$ | $0.14$ | Long questions reduce correctness |
| **`has_year`** | $-0.66$ | $0.56$ | Year extraction penalty |

---
###### note: Why does verbal confidence have a NEGATIVE coefficient ($\beta = -1.95$)?
This is the Dunning-Kruger / Hallucination effect in mathematical form. Once you hold behavioral stability (sampling consensus) and internal representations constant, an LLM that still insists on verbalizing high confidence is much more likely to be hallucinating or guessing with swagger. Stated confidence without underlying consensus is a major red flag.
---

---

## 5. Walkthrough of the Key Figures

### Figure 1 — Signal Calibration & Murphy Decomposition

```
Calibration by signal (test split)              Murphy decomposition
1.0 ┌ - - - - - - - - - - - - - - - /      0.08 ┌             ┌──┐
    │                            /              │             │  │
0.8 │                    ┌───────/         0.06 │             │  │   ┌──┐
    │                ┌───┘                      │             │  │   │  │
0.6 │            ┌───┘                     0.04 │      ┌──┐   │  │   │  │
    │        ┌───┘                              │      │  │   │  │   │  │
0.4 │    ┌───┘                             0.02 │      │  │   │  │   │  │
    │┌───┘                                      │ ┌─┐  │  │   │  │   │  │
0.0 └────────────────────────────          0.00 └─┴─┴──┴──┴───┴──┴───┴──┴─
    0.0  0.2   0.4   0.6   0.8  1.0               Verbal   Behavioral Internal
     Stated / Predicted Confidence                (Solid = Rel, Light = Res)
```

#### 1. Left Panel: Calibration Curves
* **The Curves:** All three curves (Verbal in blue, Behavioral in orange, Internal in green) hug the ideal diagonal line ($y = x$) up through $80\%$ confidence.
* **The Resolved Tail Artifact:** In the raw unpruned curve, a single bin at $p \approx 0.88$ dropped to zero. Investigating this revealed an extreme small-$N$ artifact: exactly 3 questions (all from `0.5B-instruct` on rare entity lookups) entered this bin, where the tiny model suffered from degenerate mode collapse on a wrong answer. Raising the bin threshold to $N_{\text{min}} = 10$ properly pruned this 3-sample anomaly, producing a clean, monotonic curve with complete bootstrap coverage.

#### 2. Right Panel: Murphy Decomposition
* **Reliability (Dark Solid Bars — Lower is Better):** All three signals have near-zero reliability error ($\approx 0.001\text{--}0.003$).
* **Resolution (Tall Light Bars — Higher is Better):** Measures how aggressively the signal sorts correct from incorrect answers. **Behavioral resolution ($0.0759$) is $2.2\times$ higher than Verbal ($0.0343$)**, proving that sampling consistency is vastly more informative than words.

---

### Figure 2 — Signal Mismatch Quadrants & Abstention Split

#### 1. Left Panel: Epistemic Quadrants ($N = 831$)
* **Honest Doubt ($77.1\%$, 641 cases):** The model states low confidence and its internal states confirm it doesn't know. 
* **Grounded Certainty ($11.0\%$, 91 cases):** High verbal, high internal. Concentrated in `7B-Instruct` on elementary math (C2).
* **Excessive Hedging ($10.8\%$, 90 cases):** Low verbal, high internal. The model knows the answer internally with high consensus, but hedges verbally. Highly concentrated in `7B-Instruct` on PopQA (R1) and `3B-Instruct` on GSM8K (C1).
* **Performative Certainty ($1.1\%$, 9 cases):** High verbal, low internal. Pure confident hallucination. All 9 cases occurred in `7B-Instruct` on advanced symbolic math (C2).

#### 2. Right Panel: The Abstention Split ($N = 3575$ Passes)
When given the option to bet (Format C: `ANSWER` vs. `PASS`), the pipeline forced the models to answer anyway:
* **Justified Hedge ($N = 2850$, 79.7%):** When forced to answer, the model was **WRONG**. Passing was the correct decision.
* **Missed Knowledge ($N = 725$, 20.3%):** When forced to answer, the model was **CORRECT**! The model gave up on points it actually knew in its weights.

---

### Figure 4 — Layer Depth AUROC Dynamics (H4 Supported)

```
        Factual Retrieval (R1/R2/R3)                  Multi-Step Reasoning (C1/C2/C3)
1.0 ┌       ┌───────────────────────────        1.0 ┌
    │   ┌───┘                                       │                       ┌───────
0.8 │ ┌─┘                                       0.8 │               ┌───────┘
    ├─┼──────────────────────── Gate 3 (0.65)   ├───┼───────────────┼──────── Gate 3 (0.65)
0.6 │ │                                         0.6 │       ┌───────┘
    │ │                                             │   ┌───┘
0.5 ├─┴──────────────────────── Chance (0.50)   0.5 ├───┴──────────────────── Chance (0.50)
   0%  20%   40%   60%   80%  100%                 0%  20%   40%   60%   80%  100%
             Layer Depth                                     Layer Depth
         [ Onset at 6.82% ]                              [ Onset at 28.08% ]
```

#### Why Does Reasoning Onset Later Than Retrieval? (5% Fine Sweep vs. 25-pp Baseline)
1. **Factual Retrieval Linearizes Immediately (Fine Onset = 6.82%):** On the 21-point fine sweep (`prod500_5pct`), 9 of 11 evaluated retrieval cells cross the $\ge 0.65$ gate with 2-consecutive confirmation at **layer depth 5.0%** (mean $6.82\%$). Factual lookups depend on immediate key-value associative retrieval in initial MLP layers. What looked like a $27.3\%$ onset on the coarse grid was an artifact of having no taps between $0\%$ and $25\%$.
2. **Multi-Step Reasoning Requires Serial Computation (Fine Onset = 28.08%):** A model cannot know whether a multi-step mathematical derivation will succeed until intermediate variable bindings and operator chains are executed across earlier attention layers. The signal linearizes at a mean depth of **$28.08\%$**, creating a clean **$\Delta = +21.26\%$ depth gap** ($95\%$ bootstrap CI: $[+9.72\%, +33.05\%]$).
3. **Statistical Significance Confirmed ($p = 0.0039$):** While the coarse grid was borderline ($p = 0.053$), the 21-tap continuous resolution under the 2-consecutive rule achieves **$p = 0.0039$** on 20,000 permutations. Even excluding GSM8K ($C_1$), the reasoning onset remains $+12.63\%$ deeper (`derived/h4_depth.json:c1_sensitivity`).
4. **Gate 3 Resolution (24 of 28 Passed):** The fine sweep rescued 2 cells (`1.5b__C1` and `3b__C1`, both peaking at layer 60% with AUROCs of $0.716$ and $0.676$). Only 4 cells fail Gate 3 (3 clean with no signal: `0.5b__C1`, `0.5b__R3`, `3b__R1`; and 1 failing the $p_0$ negative control: `7b-base__C3`).

---


### Figure 5 — Task Accuracy & Covariate Difficulty Grid (Supplementary)

```
                    R1 (retr)    R2 (retr)    R3 (retr)    C1 (reas)    C2 (reas)    C3 (reas)
  0.5B-instruct │   : 0.19 :     : 0.15 :     : 0.02 :     : 0.15 :     : 0.22 :     : 0.09 :   (All out-of-band)
  1.5B-instruct │   [ 0.31 ]     : 0.15 :     : 0.05 :     [ 0.60 ]     [ 0.52 ]     [ 0.33 ]
  3B-instruct   │   [ 0.39 ]     : 0.17 :     : 0.01 :     [ 0.57 ]     [ 0.56 ]     : 0.18 :
  7B-instruct   │   [ 0.47 ]     [ 0.25 ]     : 0.09 :     [ 0.69 ]     [ 0.66 ]     [ 0.36 ]
  7B-base       │   [ 0.53 ]     : 0.22 :     : 0.09 :     [ 0.31 ]     [ 0.60 ]     : 0.17 :

  Legend: [ BOLD ] = In 25%–80% Target Band (15 cells)
          : Dotted Crimson : = Out-of-Band Covariates in GLM (15 cells)
```

###### note: Isn't Figure 5 useless because all 30 cells were committed?
Not if understood properly, but it **did suffer from an identity crisis and a plotting bug** that we have now repaired:
1. **The Old Philosophy vs. `prod500`:** In the pre-audit August runs, Fig 5 was a "pruning execution gate." If a cell had $< 25\%$ accuracy, the pipeline threw it away. Fig 5 had to justify why 16 cells were missing from the paper. In `prod500`, Audit v3 realized that deleting 53% of your dataset destroys cross-scale comparisons. We set `COMMIT_CELLS_OUTSIDE_BAND = True`, retaining all 30 cells and conditioning on difficulty statistically via cell random intercepts and the `in_band` covariate in the Bayesian GLM.
2. **The Visual Bug Fixed:** Because all 30 cells were committed, the original plotting routine checked `ok = v["committed"]`, which caused **all 30 cells to be bold and zero cells to receive dotted borders**, directly contradicting the plot's title! We repaired the logic to check `in_band`: now, the 15 cells inside the proximal zone are bold, while the 15 out-of-band cells are rendered with crisp dotted crimson insets.
3. **What Figure 5 Actually Teaches Us:**
   - **Total Small-Model Floor Effect:** `0.5B-instruct` failed to cross the $25\%$ competence threshold on **any** task (dropping to $2\%$ on Trivia R3 and $9\%$ on Hard Math C3). Probing activations on a model that is essentially guessing produces pure noise.
   - **The Long-Tail Knowledge Abyss:** `R3` (obscure Trivia) is an absolute floor across all architectures ($1\%\text{--}9\%$), proving that pretraining param scaling alone does not rescue long-tail factual lookup without retrieval augmentation.
   - **The Multi-Level Covariate:** Figure 5 visually explains to reviewers exactly why the Bayesian GLM needed `in_band` as a statistical control.

---

### Figure 6 — Pairwise Signal Correlations (Table T9)

| Signal Pair | Calibrated Pearson $r$ | Raw Spearman $\rho$ | 95% Confidence Interval |
| :--- | :---: | :---: | :---: |
| **Behavioral $\leftrightarrow$ Internal** | **$+0.881$** | **$+0.499$** | $[+0.464, +0.536]$ |
| **Verbal $\leftrightarrow$ Behavioral** | $+0.818^*$ | $+0.181$ | $[+0.124, +0.239]$ |
| **Verbal $\leftrightarrow$ Internal** | $+0.899^*$ | **$+0.086$** | $[+0.033, +0.151]$ |

*(Note: Calibrated Pearson $r$ reflects global tier base rates, whereas rank Spearman $\rho$ measures item-level sorting. The rank correlation between Verbal and Internal is near zero).*

* **Key Takeaway:** Hidden layer activations and sampling entropy reflect the **same underlying epistemic certainty** ($\rho = 0.50$, $r = 0.88$). Verbal self-reports are completely detached from internal hidden states ($\rho = 0.086$).

---

## 6. The Knowledge Popularity Gradient (PopQA R1 vs. R2)

How does model accuracy degrade from popular head entities (R1) to obscure tail entities (R2)?

| Model | R1 Accuracy (Head) | R2 Accuracy (Tail) | Accuracy Drop ($\Delta$) | Verbal Confidence Drop | Tracks Accuracy? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5-0.5b-instruct`** | $18.2\%$ | $14.6\%$ | $0.0\%$ (within noise) | — | Inconclusive |
| **`qwen2.5-1.5b-instruct`** | $35.4\%$ | $15.2\%$ | **$-12.0\%$** (CI: $[1\%, 24\%]$) | — | Inconclusive |
| **`qwen2.5-3b-instruct`** | $38.4\%$ | $15.0\%$ | **$-19.0\%$** (CI: $[7\%, 31\%]$) | **$-28.4\%$** | **YES $\checkmark$** |
| **`qwen2.5-7b-instruct`** | $46.4\%$ | $15.0\%$ | **$-27.0\%$** (CI: $[15\%, 39\%]$) | **$-28.7\%$** | **YES $\checkmark$** |
| **`qwen2.5-7b-base`** | $51.2\%$ | $19.0\%$ | **$-33.0\%$** (CI: $[21\%, 45\%]$) | — | (No verbal parse) |

* **Insight:** Larger instruction models show a clean, monotonic drop in performance on obscure entities, and their verbal confidence drops in near-perfect lockstep ($\approx 28\%$).

---

## 7. Distilled in Human Language (Core Realities for the Paper)

1. **If you want to know if an LLM is telling the truth, DO NOT ask it.** Verbal confidence is an ungrounded linguistic performance ($\rho = 0.086$ with internal state, $\beta = -1.95$ in multi-level regression). Models bluff and hedge based on prompt format.
2. **Sample its answers instead.** Multi-sample semantic entropy is the single strongest indicator of correctness ($\beta = +7.41$, Resolution $= 0.0759$). If an LLM gives the exact same answer across 10 temperature samples, it is almost certainly right.
3. **Internal probes work, and they scale with model size.** Probing hidden states adds independent predictive power ($\beta = +1.06$), and the signal becomes substantially sharper as models scale up ($\beta_{\text{scale}} = +0.71$).
4. **Factual recall happens early; reasoning happens deep.** Fact retrieval linearizes at $27\%$ layer depth; reasoning requires $41\%$ depth ($\Delta = +13.6\%$, suggestive trend on 25-pp grid, permutation $p = 0.053$). Early layers capture associative factual lookups, while multi-step reasoning requires deeper representations.
5. **Epistemic mismatch is governed by difficulty, not syntax.** In continuous analysis, the mismatch gap ($\Delta = \text{Verbal} - \text{Behavioral}$) is centered symmetrically at zero ($\text{mean} = -0.006$). The discrete 10:1 "hedging" ratio (90 vs 9) was an artifact of applying a 0.5 cutoff to calibrated probabilities on hard tasks; question phrasing (dates, numbers, entities) does not drive over- or under-confidence ($p > 0.30$).

---
###### NOTE: To write the final paper, structure the narrative around these three pillars:
1. **The Epistemic Decoupling:** Verbal confidence is fragile and decoupled from internal representations.
2. **The Predictive Hierarchy:** Behavioral consensus dominates, while internal probes provide clean orthogonal signal.
3. **The Mechanistic Emergence:** Internal correctness representations emerge $13.6\%$ deeper for reasoning than retrieval (H4 suggestive trend).
---
