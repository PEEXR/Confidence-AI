# 14 Aug 1AM

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
- **The Fix**: Lines 2520–2536 of [`confidence_pipeline.py`](file:///c:/Users/systems/Documents/Confidence-AI/confidence_pipeline.py#L2520-L2536) were updated to enforce **PLAN §8.6**:
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

The (Confidence-AI/results) directory contains **6 distinct subfolders**, organizing everything from raw tensor activations to paper-ready figures and LaTeX tables:

---

### 1. ⚡ [`activations/`] — Layer Hidden States & Gate 3 Finiteness
Contains the raw neural network representations extracted during model execution:
- **`{model}__{tier}.npz`**: Compressed NumPy archives storing float32 hidden activation tensors tapped across 5 layer depth percentiles ($0\%, 25\%, 50\%, 75\%, 100\%$).
- **`{model}__{tier}.finiteness.json`**: Pre-check metadata for **Gate 3** measuring whether activations contain NaNs or numerical overflows (`nonfinite_frac`, `absmax`, `std` across percentiles `p0`–`p100`).

---

### 2. 📊 [`derived/`] — Intermediate Datasets & Hypothesis Results
Stores processed datasets (`.parquet`) and hypothesis test outputs (`.json`):
- **`signals.parquet`**: Master table joining calibrated verbal, behavioral, and internal confidence scores for every test question.
- **`graded.parquet`**: Item-level deterministic grading outcomes.
- **`entropy.parquet`**: Semantic entropy calculations over $N=10$ resampled answers.
- **`probe_sweep.parquet`**: Linear probe accuracy (AUROC) across layer depths vs. shuffle nulls and surface baselines.
- **`quadrants.parquet`**: Item-level classification into *Hopeful*, *Suppressed*, *Agree High*, and *Agree Low* confidence quadrants.

---

### 3. 🖼️ [`figures] — Publication-Ready Plots
Contains generated visual plots exported simultaneously as high-res `.png`, vector `.pdf` (for paper insertion), and `.caption.txt` (pre-registered captions):
- **`fig1_calibration`**: Calibration curves for verbal, behavioral, and internal signals.
- **`fig2_quadrant`**: Scatter plot of verbal vs. behavioral/internal confidence with quadrant splits.
- **`fig3_model_delta`**: Base vs. Instruct model contrast on hopeful confidence.
- **`fig4_depth_prediction`**: Layer depth AUROC curves (retrieval vs. reasoning).
- **`fig5_cell_commitment_grid`**: 30-cell grid heatmap showing pilot accuracy and committed cells.
- **`fig6_signal_correlations`**: Pairwise signal correlation heatmap.

---

### 4. 📄 [`tables/`] — Data Tables & Export Files
Contains all summary tables exported in three parallel formats: **`.csv`**, **`.parquet`**, and **`.tex`** (LaTeX source for direct paper inclusion):
- **`t1` – `t15`**: Tables covering dataset composition, cell commitments, accuracy, Murphy calibration decomposition, probe sweeps, depth onsets, signal correlations, and hypothesis verdicts .
- **`gate1_manual_check_sheet.csv`**: The 100-item sampling sheet generated for human verification of grading sanity.

---

### 5. ⚙️ [`meta/`] Execution Provenance & Final Report
Contains run metadata ensuring complete scientific reproducibility:
- **`final_report.json`**: Top-level executive summary of the entire run (hardware specs, measured GPU hours, gate outcomes, hypothesis verdicts).
- **`config.json`**: Full hyperparameter configuration and its 12-character SHA hash (`e9760b8e7a1d`).
- **`provenance.json`**: Platform hardware environment (RTX PRO 6000 Blackwell), PyTorch version, seed, and git commit.
- **`run_log_rows.md`**: Pre-formatted Markdown log ready for pre-registration section updates.

---

### 6. 📝 [`logs/`] — Real-Time Execution Log Stream
- **`events.jsonl`**: Structured JSON Line event stream logged continuously during notebook execution (recording cell execution times, GPU memory allocation/flushes, and progress checkpoints).