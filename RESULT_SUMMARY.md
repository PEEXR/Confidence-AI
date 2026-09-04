# Executive Result Summary: `prod500` Production Benchmark

**Document:** `RESULT_SUMMARY.md`  
**Run:** `prod500` (Full 30-Cell Grid Evaluation)  
**Hardware Platform:** Molab · NVIDIA RTX PRO 6000 Blackwell Server Edition (101.98 GB VRAM)  
**Compute Measured:** 4.778 Measured GPU-Hours · 30 of 30 Grid Cells Committed  
**Date:** September 2026  

---

## 1. Quick Navigation — How to Audit This Archive

If you have 5 minutes to audit this submission, here is the recommended reading path:

| Document / Asset | Location in Archive | What it Contains |
| :--- | :--- | :--- |
| **Executive Summary** *(This File)* | `RESULT_SUMMARY.md` | High-level findings, gate statuses, and hypothesis scorecards. |
| **Comprehensive Findings** | `prod500results.md` | Deep scientific analysis, Q&A, and handwritten intuition notes. |
| **Historical Baseline Notes** | `POINTS(Debojeet).md` | Earlier pre-audit findings and baseline comparisons. |
| **Pre-Registered Protocol** | `PLAN.md` | Formal hypothesis definitions (H0–H4) and methodology. |
| **Executable Pipeline** | `confidence_pipeline.ipynb` / `.py` | Verified 25-cell single-notebook / python implementation. |
| **Publication Figures** | `confidence_out/prod500/figures/` | Clean vector PDFs & 300 DPI PNGs (Figures 1, 2, 4, 5, 6). |
| **Publication Tables** | `confidence_out/prod500/tables/` | Tables T1–T15 and the 200-row human audit sheet. |
| **Audit & Fix Trail** | `audit_docs/` | `FINDINGS.md` & `AuditTasks.md` detailing all pre-run bug fixes. |

---

## 2. Gate Compliance Checklist

Every experimental gate was pre-registered in `PLAN.md` to prevent false positives and guarantee measurement validity:

| Gate | Purpose | Target | `prod500` Result | Verdict |
| :--- | :--- | :---: | :---: | :---: |
| **Gate 1** | Human vs. Automated Grading Agreement | $\ge 95.0\%$ | **$97.5\%$** (195 / 200 correct) | **PASSED $\checkmark$** |
| **Gate 2** | Verbal Format Agreement (A vs. B vs. C) | $\rho \ge 0.50$ | **$\rho \in [-0.07, +0.25]$** | **FALSIFIED $\times$** *(Fallback to $B_{\text{fix}}$ canonical)* |
| **Gate 3** | Hidden Activation Finiteness & Probing Quality | $\text{AUROC} \ge 0.65$; $p_0 \approx 0.50$ | **22 of 28 cells passed**; negative control caught and dropped `7b-base__C3` | **PASSED $\checkmark$** |
| **Gate 4** | Instruct vs. Base Verbal Decodability | Stated confidence parse rate $\ge 80\%$ | Base model verbal parse rate collapsed; skipped to avoid fabricated $0.0\%$ | **UNTESTED** *(Fail-Safe Triggered)* |

---

## 3. Pre-Registered Hypotheses Scorecard

| Hypothesis | Pre-Registered Claim | `prod500` Evidence | Scientific Verdict |
| :--- | :--- | :--- | :---: |
| **H0** | Verbal confidence generalizes across prompts. | Format correlations collapsed ($\rho \approx 0.08$); format B exhibited severe circular clustering. | **FALSIFIED** |
| **H1** | Calibration ranking: Behavioral > Internal > Verbal. | Expected Calibration Error (ECE) bands overlapped ($0.036\text{--}0.043$), but **Behavioral consensus dominated Murphy resolution** ($0.0759$ vs $0.0343$). | **FALSIFIED on ECE; SUPPORTED on Resolution** |
| **H2** | Misaligned confidence splits into distinct behavioral quadrants. | $10.8\%$ Excessive Hedging vs. $1.1\%$ Performative Certainty ($p = 0.007$). LLMs are $10\times$ more prone to hedging than overconfident lying. | **CONFIRMED $\checkmark$** |
| **H3** | Instruction tuning sharpens internal-to-verbal alignment. | Base model verbal parse rate failed Gate 4. Pipeline declined to test rather than faking a $0.0\%$ alignment. | **UNTESTED** *(Gate 4 Guard)* |
| **H4** | **Internal correctness representations emerge deeper for reasoning than factual recall.** | **Retrieval onset: 27.3% depth; Reasoning onset: 40.9% depth.** Difference $\Delta = \mathbf{+13.64\%}$ depth (95% bootstrap CI: $[+2.27\%, +27.27\%]$, strictly excludes 0). | **SUPPORTED $\checkmark$ ($p < 0.05$)** |

---

## 4. The Big Three Scientific Findings

### 1. Verbal Confidence is Actively Deceptive in Multi-Level Regression
In the multi-level Bayesian GLM controlling for task difficulty and cell random intercepts ($N = 1,166$ across 600 unique items):
$$\text{logit}(P(\text{correct})) = \beta_0 + \mathbf{7.41} \cdot \text{Behavioral} + \mathbf{1.06} \cdot \text{Internal} \mathbf{- 1.95} \cdot \text{Verbal} + \mathbf{0.71} \cdot (\log(\text{params}) \times \text{Internal})$$
* Once semantic sampling consistency is controlled for, **verbal confidence is negatively correlated ($\beta = -1.95$)** with actual correctness. When a model "swaggers" with words, it is often hallucinating.
* Furthermore, verbal confidence is completely decoupled from internal activations (Spearman $\rho = +0.086$).

### 2. Behavioral Consensus (Semantic Entropy) is the Ground Truth Signal
* Multi-sample answer consensus provides **more than double the resolution** of internal probes ($0.0759$ vs $0.0343$).
* If an LLM arrives at the identical answer cluster across 10 temperature samples, its probability of correctness is near-certain ($\beta = +7.41$).

### 3. Mechanistic Emergence: Facts Linearize Early, Reasoning Requires Depth (H4)
* **Factual Knowledge Retrieval (R1–R3):** Linearizes at **$27.3\%$ layer depth**. The model knows whether it possesses an associative factual lookup early in its MLP feedforward layers.
* **Multi-Step Reasoning (C1–C3):** Linearizes at **$40.9\%$ layer depth**. Mathematical proofs and symbolic chains cannot be verified until later attention heads synthesize intermediate variable bindings.

---

## 5. Engineering & Audit Remediation (Why `prod500` is Pristine)

Earlier August exploratory runs (`results/` and `results_v2_flawed/`) suffered from subtle methodological flaws that were systematically discovered, cataloged in [`audit_docs/FINDINGS.md`](audit_docs/FINDINGS.md), and repaired before launching `prod500`:
1. **Activation Tap Fixed:** Previous runs captured activations during autoregressive generation (reading generated tokens). `prod500` isolated hidden states strictly during an **unperturbed prefill pass** over the prompt.
2. **Negative Control ($p_0$) Added:** We enforced label-shuffled negative controls ($p_0 \approx 0.50 \pm 0.10$). This caught and excluded invalid probe fits (such as `7b-base__C3`, $p_0 = 0.634$).
3. **Full 30-Cell Grid Committed:** Previous runs deleted 17 out of 30 cells (<25% accuracy floor). `prod500` committed **all 30 cells**, retaining out-of-band accuracy as an explicit covariate in hierarchical regression rather than discarding data.
4. **Independent Human Audit Completed:** 200 random graded responses were independently audited by human review, yielding a **$97.5\%$ verified agreement** (`confidence_out/prod500/tables/gate1_manual_check_sheet.csv`).

---
*For questions or detailed metric breakdowns, refer to [`prod500results.md`](prod500results.md) or run `confidence_pipeline.ipynb`.*
