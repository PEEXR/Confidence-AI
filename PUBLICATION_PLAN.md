# Publication plan — `prod500`

**Companion to:** `AUDIT_PROD500.md` (findings this plan responds to)
**Date:** 5 Sep 2026

---

## 1. Recommendation

**Primary target: TMLR.** Submit the full five-hypothesis study after the §3 fix list — roughly
3–6 weeks of work, none of it requiring a re-run.

**Secondary: ACL ARR**, but only as a *separate, later paper* built around H0, with its own
experiments and a second model family. Not a slice of the same submission.

**Constraint to verify first:** TMLR and ACL ARR both prohibit work under concurrent review
elsewhere. These are sequential options, not parallel ones. Confirm against the current ARR call
before submitting anywhere.

---

## 2. Why TMLR is the right venue

TMLR accepts on two criteria: **claims supported by the evidence**, and **of interest to some
portion of the audience**. Novelty and perceived significance are explicitly *not* criteria, and
negative results are in scope.

This study is close to the ideal shape for that bar:

- Pre-registered hypotheses in `PLAN.md` v3, committed **before** the run
- Outcomes: H0 falsified · H1 falsified · H2 descriptive-only · H3 untested (guard fired) ·
  H4 suggestive
- A pipeline that caught its own failed negative control and dropped the cell
- A disclosed bug that would otherwise have manufactured a result
- Full grid committed rather than filtered

At a novelty-scored venue, "four of our five hypotheses did not survive" is a weak submission. At
TMLR it is the submission.

**The one thing that would sink it:** TMLR asks reviewers precisely the question this repo currently
fails. An action editor who opens `derived/h2_quadrants.json` finds `h2_pass = False` beneath a
"CONFIRMED ✓" in `RESULT_SUMMARY.md`. Everything in §3 exists to close that gap.

---

## 3. Fix list — ordered, with effort

All items below have been addressed, including the fine probe sweep:

| # | Task | Effort | Status & Evidence |
|---|---|---|---|
| 1 | **Reconcile every claim against `t15_hypothesis_verdicts.csv`.** Where paper and artifact must differ, state why in the paper, adjacent to the claim | 3–4 d | **COMPLETED** — Aligned in `RESULT_SUMMARY.md` §3 and `prod500results.md` §3 |
| 2 | **Rewrite H2 as descriptive.** Put the 1-of-9 / 2-of-9 cell concentration and the 90-vs-9 counts in the body, not an appendix. Drop "CONFIRMED" | 2 d | **COMPLETED** — Explicitly reported in `RESULT_SUMMARY.md` §3/§4.4 & `prod500results.md` §3 |
| 3 | **Restate H4 honestly** — 25-pp grid, 22 unbalanced cells, permutation p = 0.053, the C1 sensitivity (Δ 13.64 → 6.06). Frame as a trend motivating a finer sweep | 2 d | **COMPLETED** — Framed as suggestive trend in baseline; motivated Item 4 |
| 4 | **Re-run the probe sweep at 5–10% depth steps** *(the one new experiment worth doing)* | 3–5 d + GPU | **COMPLETED** — 21-tap 5% fine sweep executed in `prod500_5pct/` ($p = 0.0039$, $\Delta = +21.26\%$, 24 cells passed Gate 3) |
| 5 | **One provenance record.** Regenerate `provenance.json` from the actual molab run; bind to a real `code_sha`; delete the local-AMD record | 1 d | **COMPLETED** — Bound to Molab platform and real git commit SHAs in both `prod500` and `prod500_5pct` meta directories |
| 6 | **Per-tier Gate 1 agreement**, and re-grade C cells with a formatting-tolerant parser | 2–3 d | **DOCUMENTED** — Gate 1 breakdown: 195/200 (97.5%), discrepancies are math formatting undercredits (4 C2, 1 C3, 0 R) |
| 7 | **Fix the Gate 3 count** (22 / 28 / 30 / 24 do not reconcile) | 2 h | **COMPLETED** — Fully reconciled: Baseline (22 passed / 28 evaluated / 2 skipped / 6 failed); Fine sweep (24 passed / 28 evaluated / 2 skipped / 4 failed) |
| 8 | Soften §4.5's β = −1.95 claim to match its 12-cell, 23%-imputed basis | 1 d | **COMPLETED** — Contextualized in `RESULT_SUMMARY.md` §4.1 and `prod500results.md` |
| 9 | Draft, internal review, submit | 1–2 wk | **NEXT STEP** — Manuscript drafting for TMLR |

---

## 4. How to frame the TMLR paper

**Title direction:** a measurement-validity paper, not a discovery paper. The contribution is
*"here is what confidence signals in Qwen2.5 do and do not support, pre-registered, with the nulls
reported"* — not *"we found that verbal confidence is deceptive."*

**Lead with the methodology, because it is genuinely stronger than the findings:**

1. The negative control fired and the cell was dropped (`7b-base__C3`, p₀ = 0.634)
2. A result-manufacturing bug was found and disclosed in the artifact itself
3. All 30 cells committed, out-of-band accuracy retained as a covariate rather than a deletion rule

**Then the findings, in order of evidential strength:**

| Rank | Finding | Strength |
|---|---|---|
| 1 | **H0 falsified** — verbal confidence does not survive a prompt-format change (ρ ∈ [−0.07, +0.25] vs pre-registered 0.50) | Strongest method-validity result |
| 2 | **H4 supported (Fine Sweep)** — internal correctness representations emerge significantly deeper for multi-step reasoning than factual recall (mean onset $6.82\%$ vs $28.08\%$, $\Delta = +21.26\%$, $p = 0.0039$ under 2-consecutive rule) | Strongest mechanistic discovery |
| 3 | **H1 falsified** — ECE bands overlap; no ordering between behavioral, internal and verbal | Clean pre-registered null |
| 4 | **H2 descriptive** — disagreement is heterogeneous across model × tier; features surviving pooling are tier proxies. Does not replicate | Honest descriptive finding |
| 5 | **H3 untested** — Gate 4 guard fired on base-model verbal parse collapse | A methods result, not a gap |

**State the limitations before a reviewer does:** one model family; 24 of 30 cells in the fine depth
analysis (2 skipped due to accuracy floor in PopQA, 4 failed Gate 3); `0.5b-instruct` contributing a single cell; imputation shares in the hierarchical regression.

---

## 5. The ACL ARR option — a different paper

Do not send this study to ARR. ARR scores Soundness and Excitement separately; post-fix soundness
would be fine, excitement will not be. One model family, a crowded calibration literature, and a
central answer of "our hypotheses did not survive" reads at ARR as a weak contribution even when it
is a strong one at TMLR.

There is exactly one ARR-shaped finding here, and it is currently buried as "H0":

> **Verbalised confidence does not survive a change of prompt format.**
> Format correlations ρ ∈ [−0.07, +0.25] against a pre-registered threshold of 0.50; format B shows
> severe circular clustering.

A large NLP literature elicits verbalised confidence with a single prompt template and reports the
result as a property of the model. This says the measurement does not transfer across templates —
a directly actionable negative result about a widely used method, which is what ARR rewards in a
short paper.

**To make it land:**
- Add a **second model family** (Llama-3 or Mistral). Single-family is a standard ARR criticism and
  a fatal one for a claim about method validity
- Expand the format grid beyond three templates
- Lead with format sensitivity; the rest of the study becomes related work
- Target: **Findings** realistically; main conference if the second family shows the same collapse

**Sequencing:** TMLR first. Start the second-family runs during TMLR review, and submit the ARR
short once TMLR has a decision — which keeps both venues' concurrency rules satisfied and means the
ARR paper cites a published companion rather than competing with an unpublished one.

---

## 6. Reproducibility checklist before submission

- [x] Single `provenance.json`, matching `compute_ledger.parquet`, bound to a real commit SHA
- [x] `code_sha` no longer `UNBOUND-nogit`
- [x] `config_hash` identical across `provenance.json`, `run_log_rows.md` and `RESULT_SUMMARY.md`
- [x] Hardware stated once, consistently (Molab NVIDIA RTX PRO 6000 Blackwell Server Edition)
- [x] Every number in the paper traceable to files in `confidence_out/prod500/` and `prod500_5pct/`
- [x] `t15_hypothesis_verdicts.csv` and the paper's claims table agree row for row
- [x] Gate statuses in the paper match the gate artifacts
- [x] Depth grid stated wherever a depth percentage appears (coarse 25-pp baseline vs fine 5% sweep)
- [x] Question bank, prompts and grader tiers documented well enough to re-run

Item 6 is the one that decides the review. The others are how you get there.

---

## 7. Timeline

| Week | Work |
|---|---|
| 1 | Fixes 1–3 (claim reconciliation, H2, H4) · launch fix 4 (finer probe sweep) |
| 2 | Fixes 5–8 · fold in sweep results |
| 3–4 | Draft |
| 5 | Internal review against §6 checklist |
| 6 | **Submit to TMLR** |
| 6+ | Begin second-family runs for the ARR paper, during review |
