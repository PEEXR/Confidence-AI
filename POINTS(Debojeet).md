# 14 Aug

---
###### NOTE: ALL "NOTES" ARE HANDWRITTEN AND ARE IMPORTANT
TO DO: go through tables and derived, Human grading sanity check is still left
CURRENTLY : Understanding figures
UNDERSTOOD: Eval Methods, hypotheses
---

## differences between results, results prepair and resultsv2flawed

The three directories represent **three sequential iterations** of the experimental pipeline, tracing the progression from an incomplete run to a flawed run, and finally to the clean, authoritative dataset:

---

### Comparison Summary

| Feature / Run | `results_prerepair/` (Run 1) | `results_v2_flawed/` (Run 2) | `results/` (Final Clean Run) |
| :--- | :--- | :--- | :--- |
| **Run Date** | Aug 12, 20:25 UTC | Aug 13, 02:00 UTC | Aug 13, 06:09 UTC |
| **Committed Cells** | 10 of 30 cells | 13 of 30 cells | **13 of 30 cells** |
| **7B-Base Status** | Incomplete (0.13 GPU-hr) | Completed (1.52 GPU-hr) | **Completed (1.52 GPU-hr)** |
| **H3 Evaluation** | Untested (Missing 7B-Base) | Evaluated | **Evaluated** |
| **H1 Verdict** | Falsified | **False Positive** (*"Supported"*) | **Falsified** *(Correct)* |
| **Status** | ❌ Incomplete run | ❌ Flawed calibration logic | ✅ **Final Authoritative Output** |

---

### Detailed Breakdown

#### 1. `results_prerepair/` (Initial Incomplete Run)
- **What happened**: The first full-grid attempt was interrupted before `qwen2.5-7b-base` could complete its evaluation cells (only logging 0.13 GPU-hours).
- **Impact**: Only 10 cells were committed, and **H3 could not be tested** because comparing Instruct vs. Base required both 7B models to finish.

#### 2. `results_v2_flawed/` (Completed, but Methodologically Flawed)
- **What happened**: All 5 models and 13 cells finished execution. However, it contained a critical bug in how **verbal signal calibration (H1)** was handled.
- **The Flaw**: When a model's verbal outputs degenerated into choosing a single constant value (e.g. constant 90% confidence), that cell was flagged as degenerate (`< MIN_DISTINCT_VERBAL`), but the flag was ignored during scoring. The pipeline mistakenly assigned this constant predictor an $ECE \approx 0.0$, falsely reporting that verbal confidence was "better calibrated than internal signals."

#### 3. `results/` (Final, Authoritative Benchmark Output)
- **The Fix**: Lines 2520–2536 of [`confidence_pipeline.py`] were updated to enforce **PLAN §8.6**:
  > *"A cell whose verbal signal has fewer than `MIN_DISTINCT_VERBAL` distinct values is EXCLUDED from the verbal comparison — not reported as 'well calibrated'."*
- **Outcome**: Degenerate cells were properly excluded, revealing the true statistical reality: confidence intervals for all three signals overlap ($ECE \approx 0.028\text{--}0.040$), resulting in the corrected **"H1 Falsified"** verdict.

---

> [!IMPORTANT]
> Always use **`results/`** for all reports, paper figures, and data analysis. `results_prerepair/` and `results_v2_flawed/` are historical run logs kept for transparency and provenance audit.

## Are all the correleations calculated separately for each dataset tier, or all together?

The headline correlations for H0 / Gate 2 ($\rho_{A-B} \approx 0.217$, $\rho_{A-C} \approx -0.047$, $\rho_{B-C} \approx 0.050$) were calculated all together (pooled) across all dataset tiers and models in the pre-registered agreement subset.

However, the pipeline also computed per-model breakdowns in [h0_gate2.json]

Even at the individual model level, no format pair reliably achieved the $\rho \ge 0.60$ threshold (the highest observed was $\rho_{A-C} = 0.432$ on 7B-Instruct), validating that the disagreement between verbalized formats is a fundamental property rather than an artifact of pooling across model scales.

---
###### NOTE: could try per dataset correlations in future
---

## do these formats align exactly with the 3 way confidence of PLAN.md?


| Model Variant | Pair A–B ($\rho$) | Pair A–C ($\rho$) | Pair B–C ($\rho$) |
| :--- | :---: | :---: | :---: |
| **Qwen2.5-0.5B-Instruct** | — | $-0.008$ ($n=181$) | — |
| **Qwen2.5-1.5B-Instruct** | $-0.043$ ($n=269$) | $-0.032$ ($n=506$) | $+0.398$ ($n=254$) |
| **Qwen2.5-3B-Instruct** | $+0.307$ ($n=300$) | $+0.181$ ($n=599$) | $+0.215$ ($n=300$) |
| **Qwen2.5-7B-Base** | $+0.106$ ($n=300$) | $-0.090$ ($n=596$) | $+0.008$ ($n=300$) |
| **Qwen2.5-7B-Instruct** | $+0.268$ ($n=399$) | $+0.432$ ($n=599$) | $-0.0002$ ($n=399$) |

Formats A, B, and C align 100% with PLAN §4 as the pre-registered formats for measuring Signal 1 (Verbalized Confidence).
The Pre-registered Gate 2 Rule: PLAN §4 required checking whether Formats A, B, and C agreed before picking a single "canonical" verbal format. Because H0 showed they did not correlate ($\rho < 0.6$), the pipeline triggered the pre-registered Gate 2 Fallback, reporting all 3 formats separately against Behavioral (Signal 2) and Internal (Signal 3) confidence.

---
###### note: only 13 cells were committed, could try with higher tier models with better maths solving capabilities, looks like simpleQA and complex-math is too hard for these models.
---

---
###### note: Human grading sanity check is still left, will be done asap
---

## How to read the results?

The (Confidence-AI/results) directory contains **6 distinct subfolders**, organizing everything from raw tensor activations to paper-ready figures and LaTeX tables:

---

### 1. [`activations/`] — Layer Hidden States & Gate 3 Finiteness
Contains the raw neural network representations extracted during model execution:
- **`{model}__{tier}.npz`**: Compressed NumPy archives storing float32 hidden activation tensors tapped across 5 layer depth percentiles ($0\%, 25\%, 50\%, 75\%, 100\%$).
- **`{model}__{tier}.finiteness.json`**: Pre-check metadata for **Gate 3** measuring whether activations contain NaNs or numerical overflows (`nonfinite_frac`, `absmax`, `std` across percentiles `p0`–`p100`).

---

### 2. [`derived/`] — Intermediate Datasets & Hypothesis Results
Stores processed datasets (`.parquet`) and hypothesis test outputs (`.json`):
- **`signals.parquet`**: Master table joining calibrated verbal, behavioral, and internal confidence scores for every test question.
- **`graded.parquet`**: Item-level deterministic grading outcomes.
- **`entropy.parquet`**: Semantic entropy calculations over $N=10$ resampled answers.
- **`probe_sweep.parquet`**: Linear probe accuracy (AUROC) across layer depths vs. shuffle nulls and surface baselines.
- **`quadrants.parquet`**: Item-level classification into *Hopeful*, *Suppressed*, *Agree High*, and *Agree Low* confidence quadrants.

---

### 3. [`figures] — Publication-Ready Plots
Contains generated visual plots exported simultaneously as high-res `.png`, vector `.pdf` (for paper insertion), and `.caption.txt` (pre-registered captions):
- **`fig1_calibration`**: Calibration curves for verbal, behavioral, and internal signals.
- **`fig2_quadrant`**: Scatter plot of verbal vs. behavioral/internal confidence with quadrant splits.
- **`fig3_model_delta`**: Base vs. Instruct model contrast on hopeful confidence.
- **`fig4_depth_prediction`**: Layer depth AUROC curves (retrieval vs. reasoning).
- **`fig5_cell_commitment_grid`**: 30-cell grid heatmap showing pilot accuracy and committed cells.
- **`fig6_signal_correlations`**: Pairwise signal correlation heatmap.

---

### 4. [`tables/`] — Data Tables & Export Files
Contains all summary tables exported in three parallel formats: **`.csv`**, **`.parquet`**, and **`.tex`** (LaTeX source for direct paper inclusion):
- **`t1` – `t15`**: Tables covering dataset composition, cell commitments, accuracy, Murphy calibration decomposition, probe sweeps, depth onsets, signal correlations, and hypothesis verdicts .
- **`gate1_manual_check_sheet.csv`**: The 100-item sampling sheet generated for human verification of grading sanity.

---

### 5. [`meta/`] Execution Provenance & Final Report
Contains run metadata ensuring complete scientific reproducibility:
- **`final_report.json`**: Top-level executive summary of the entire run (hardware specs, measured GPU hours, gate outcomes, hypothesis verdicts).
- **`config.json`**: Full hyperparameter configuration and its 12-character SHA hash (`e9760b8e7a1d`).
- **`provenance.json`**: Platform hardware environment (RTX PRO 6000 Blackwell), PyTorch version, seed, and git commit.
- **`run_log_rows.md`**: Pre-formatted Markdown log ready for pre-registration section updates.

---

### 6. [`logs/`] — Real-Time Execution Log Stream
- **`events.jsonl`**: Structured JSON Line event stream logged continuously during notebook execution (recording cell execution times, GPU memory allocation/flushes, and progress checkpoints).

# 15 AUG

---
###### note: interesting resultss in depth figures. reasoning peaks at early layers for some reason. not only that, reasoning is useless on 7b base model, much worse than even much smaller instruct models?? idk if this is weird behaviour, need help to make sense
---


## What the results shoW?

### 1. Factual Retrieval and Reasoning Form at the Same Layer Depth (~25%)
- **Pre-Registered Prediction (H4)**: It was hypothesized that factual retrieval (PopQA R1) would appear at layer 0%, while multi-step reasoning (GSM8K C1, MATH C2) would only emerge in late layers (75%–100%).
- **What Figure 4 Proves**: **Both retrieval and reasoning confidence emerge at the exact same depth (~25% of total model layers)**.
  - At **0% depth** (input token embeddings), AUROC is at chance ($\sim 0.50$).
  - By **25% depth**, AUROC jumps sharply across both retrieval ($\approx 0.81\text{--}0.83$) and reasoning ($\approx 0.70\text{--}0.80$).
  - Mean onset for Retrieval is **$18.75\%$** vs **$21.43\%$** for Reasoning ($\Delta = +2.68\%$, CIs overlap). This falsifies H4, showing that internal knowledge of correctness is computed early across all task types.

---

### 2. Retrieval Achieves a Much Higher Predictability Ceiling Than Reasoning
While both start early, their asymptotic accuracy ceilings differ significantly:
- **Retrieval (PopQA R1)** reaches high asymptotic AUROCs of **$0.85 \text{ to } 0.91$**.
- **Reasoning (GSM8K C1, MATH C2/C3)** plateaus noticeably lower, between **$0.67 \text{ to } 0.80$**.
- **Takeaway**: Internal hidden activations can predict whether the model knows a factual entity with near certainty ($91\%$), but predicting whether multi-step mathematical reasoning will succeed is intrinsically noisier and harder for a linear probe to decode.

---

### 3. Where Do Signals Peak? (Peaking Dynamics)
- **Reasoning tasks (C1/C2/C3)** often peak **early-to-midway (25% to 50% depth)**:
  - `1.5B C1`: Peaks at **25% depth** ($0.766$ AUROC) $\rightarrow$ drops to $0.716$ at 100%.
  - `1.5B C2`: Peaks at **25% depth** ($0.801$ AUROC) $\rightarrow$ drops to $0.755$ at 100%.
  - `7B-Instruct C1`: Peaks at **50% depth** ($0.757$ AUROC) $\rightarrow$ drops to $0.705$ at 100%.
  - `7B-Instruct C3`: Peaks at **25% depth** ($0.669$ AUROC) $\rightarrow$ drops to $0.556$ at 100%.
- **Retrieval tasks (R1)** continue accumulating signal and peak in **late layers (75% depth)**:
  - `1.5B R1`: Peaks at **75% depth** ($0.865$ AUROC).
  - `3B R1`: Peaks at **75% depth** ($0.911$ AUROC).
  - `7B-Instruct R1`: Peaks at **75% depth** ($0.850$ AUROC).
- **Takeaway**: Reasoning confidence is crystallized in early-mid computation layers; later layers focus on formatting and generation tokens, which slightly disperses the linear probe signal.

---

### 4. Base Models Completely Lack Linear Reasoning Representations
Comparing `qwen2.5-7b-base` vs. `qwen2.5-7b-instruct`:
- **On Retrieval (R1)**: The Base model has strong probe signal ($0.835$ AUROC at 75% depth), showing raw pre-trained weights store factual knowledge cleanly.
- **On Reasoning (C1/C2)**: The Base model **fails completely** ($0.50\text{--}0.58$ AUROC, failing Gate 3 and failing to beat shuffle nulls).
- **Takeaway**: Post-training (SFT / RLHF) is what organizes internal hidden states into linearly separable trajectories during multi-step reasoning. Without instruction tuning, internal probes cannot read reasoning certainty.

---

### 5. Scaling Model Size Does Not Shift Depth Onset Earlier
- Scaling from **1.5B $\rightarrow$ 3B $\rightarrow$ 7B** does not compress relative onset depth (Spearman $\rho = 0.08, p = 0.82$).
- **Takeaway**: Relative computation depth is scale-invariant; larger models use the same proportion of their transformer depth (~25%) to form initial certainty estimates.

---

### Summary Table for Note-Taking

| Aspect | Retrieval (PopQA R1) | Reasoning (GSM8K / MATH) |
| :--- | :--- | :--- |
| **Depth Onset** | Emerges early (**~25% depth**) | Emerges early (**~25% depth**) |
| **Peak Layer** | Late layers (**~75% depth**) | Early-mid layers (**25% – 50% depth**) |
| **Max AUROC** | Very high (**0.85 – 0.91**) | Moderate (**0.67 – 0.80**) |
| **Base Model Signal** | Strong ($0.835$) | Non-existent ($0.506$ – $0.588$) |
| **Scaling Effect** | Scale-invariant relative depth | Scale-invariant relative depth |

---
###### NOTE: Format A was chosen for verbalized. perhaps its worth rerunning these experiments with format B or C?
---

## usee of CI
### 4. How 95% CIs Determine Pass/Fail Verdicts in this Study

In pre-registered hypothesis testing, 95% CIs are used to make statistical decisions without relying on arbitrary p-values:

| Hypothesis / Gate | Rule Using 95% CI | Observed Result | Verdict |
| :--- | :--- | :--- | :---: |
| **Gate 2 (H0 - Format Agreement)** | Lower bound of CI must be $\ge +0.60$ for all format pairs. | $A\text{--}C$ CI: $[-0.087, -0.006]$ (Lower bound $< 0.60$) | **Failed / Falsified** |
| **H1 (Signal Separability)** | The $95\%$ CI of $\Delta(\text{worst} - \text{best ECE})$ must **exclude zero**. | $\Delta$ CI: $[-0.014, +0.034]$ (**Includes zero**) | **Falsified** |
| **H3 (Base vs Instruct Delta)** | $95\%$ CI for change in "hopeful" rate must **exclude zero** AND clear the missed-knowledge guard. | $\Delta$ CI: $[-0.272, -0.205]$ (Excludes zero, but guard failed) | **Null** |

##    the 5 hypotheses

### Summary Card

| Hypothesis | Predicted Effect | Observed Reality | Verdict |
| :--- | :--- | :--- | :---: |
| **H0** | Formats A/B/C correlate ($\rho \ge 0.60$) | Formats do not correlate ($\rho \approx -0.05 \text{ to } +0.22$) | **Falsified** |
| **H1** | Verbalized has worse ECE than internal/behavioral | All 3 signals have similar calibration ($ECE \approx 0.03$) | **Falsified** |
| **H2** | Disagreements cluster by question type | Long & multi-entity questions strongly drive "hopeful" states ($p < 10^{-5}$) | **SUPPORTED $\checkmark$** |
| **H3** | Instruct creates hopeful confidence with clean guard | Instruct is hopeful ($23.8\%$), but missed-knowledge guard fired | **Null** |
| **H4** | Retrieval starts at 0%; Reasoning starts late | Both emerge at ~25% depth; no scale compression | **Falsified** |

---
###### note: Format C has 2 distinct values (ANSWER or PASS), which the pipeline maps to exactly two point probabilities:

$$\text{ANSWER} \implies p = \frac{1 + 2/3}{2} \approx 0.833$$ $$\text{PASS} \implies p = \frac{2/3}{2} \approx 0.333$$

Because of this, Format C’s $n_{\text{distinct}}$ is structurally capped at 2 and can never reach 3, failing the pre-flight criterion (`MIN_DISTINCT_VERBAL >= 3`).  Is this the error manan bhaiya was talking about?

---

## Why A Is the Best-Calibrated Verbal Format
Information Density: Format A preserves the model's fine-grained internal confidence differences rather than crushing them into 5 coarse buckets (B) or 2 betting actions (C).
Robustness to Degeneracy: Format A avoids the degenerate constant-predictor collapse that plagues empirical bucket mapping in Format B.
Calibrator Fit: Standard non-parametric calibrators (isotonic regression) require varied inputs across $[0, 1]$ to build an accurate reliability diagram, which only Format A reliably provides.

## Error in methodology?

### 1. The Theoretical Argument

Conceptually, **Format A (Numeric 0–100%)** is the only format designed to provide:
1. **Continuous expressivity**: A true continuous spectrum across $[0, 1]$ rather than 5 discrete words.
2. **True Metacognition**: Forcing the model to output fine-grained probabilities rather than linguistic shortcuts.
3. **Proper Isotonic Fit**: Non-parametric calibrators work best when fed continuous numeric inputs.

In theory, Format A *should* be the cleanest, most expressive verbal confidence signal.

---

### 2. What the Code's Automated Metric Did (`confidence_pipeline.py`)

When the code executed Stage 10 (`assemble_signals`), it had to automatically choose the canonical format based on an objective formula ([`confidence_pipeline.py:L2470`])
$$\text{canonical} = \arg\min_{\text{eligible}} (\text{Brier Score on Calibration Split})$$

Look at what happened when the code computed the Brier scores ([`calibration_meta.json`])
```
  • Format A (Raw Numeric):   Brier = 0.523  (High error because raw LLMs are uncalibrated)
  • Format B (Bucket Map):    Brier = 0.219  (Lower error because buckets are mapped to accuracy)
```

Because **$0.219 < 0.523$**, the Python script automatically assigned:
```python
canonical = "B"
```

---

### 3. The Consequence (Why This is the Core Takeaway for the Paper)

This is the central finding:
* **The algorithm picked Format B** because empirical bucket mapping artificially deflated its Brier score on the training/calibration set.
* **BUT once Format B was deployed into individual cells**, the fundamental weakness of bucket mapping surfaced: **7 out of 13 cells collapsed into outputting a single bucket (`n_distinct < 3`)**, forcing PLAN §8.6 to exclude them!

### Summary
* **Theoretically**: Format A is the most robust construct for continuous calibration.
* **Operationally in the pipeline**: Format B was automated as canonical because its empirical mapping scored a lower calibration Brier score ($0.219$), which subsequently caused the 7-cell mode collapse.

---
###### NOTE:Behavioral achieved the best overall Brier score of $0.166$ because it had by far the highest Resolution ($0.0787$), meaning it was best at actually separating right answers from wrong answers. check fig1.
###### just for reference, ece and brier calculated for ABC, internal behavioral and verbal, for 32 cells (7 degenerate cuz n_distinct < 3)
---
