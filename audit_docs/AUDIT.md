# Confidence-AI (Genuine vs. Hopeful Confidence) — audit

**Repo:** github.com/PEEXR/Confidence-AI
**Claim:** Measure LLM confidence three ways on the same questions — verbalized (3 formats),
behavioral (semantic entropy over N=10), internal (hidden-state probe at 5 depths) — and
characterize the disagreement quadrants: "hopeful"/performed vs "suppressed" confidence.

| | |
|---|---|
| **Current level** | **L1 — instrument mostly built, headline claims must be withdrawn** |
| **Expected level** | **L3** (Findings / TMLR) as a measurement-validity paper |
| Code trust | 5/10 |
| Results reliability | 4/10 |
| Methodology | 4/10 |
| Hypothesis | 4/10 |
| Novelty | Incremental; stated headline scooped |
| **Completion** | **~25% of planned grid** |

---

## What the audit cleared

**The quadrant thresholds are not gamed.** `QUADRANT_THRESHOLD = 0.5` is a fixed
constant; there is no median or quantile anywhere in the path. This was the most likely
way the finding could have been definitional, and it isn't. (PLAN.md specifies no number,
so 0.5 is post-hoc — but post-hoc is not the same as data-fitted.)

**The quadrant analysis exists** — `h2_quadrants.json`, n=1199, 224 hopeful, 169
suppressed. This is not "an instrument with no finding."

**Pre-registration timing holds.** PLAN.md v3 committed 2026-08-12; all three result trees
first appear 2026-08-13. *(Verified by orchestrator against full history — a subagent
initially reported this unverifiable due to a shallow-clone artifact on my side.)*

**The v2 repair was honest.** Excluding verbal degeneracy cost 53% of n and **flipped H1
from supported to falsified**. The team took the hit rather than hiding it.
`nonfinite_frac = 0.0` verified across all 65 shards. No SMOKE contamination.

## Blocking findings

### 1. The activation tap reads the wrong token — decisive
`tap.pop()` is called **after** `generate()`, so the hook's buffer holds the **last
generated token**, not the last prompt token.

This contradicts `CHANGESforPLANv3.md` §4 verbatim — "pre-generation depth-of-signal",
"explicitly not doing" during-generation probing. Confirmed empirically: 7B-base `p0`
scores **AUROC 0.717 on a token that must be constant**.

Every internal-probe number in the repo measures something other than what the design
document specifies.

### 2. The verbal axis is label-derived
The canonical format is **B**, whose raw value *is the empirical accuracy of the bucket
the model chose*, fit on the calibration split — then calibrated again on that same split.
`format_stats.B.ece = 2.3e-17` — identically zero by construction.

So "high verbalized confidence" is not what the model said; it is a supervised P(correct)
estimate wearing the model's label. This is what makes the three-signal comparison
non-commensurable (below).

### 3. Semantic entropy is a shortcut
`microsoft/deberta-large-mnli` is the correct model (Kuhn's own) and bidirectional
entailment + union-find is right. But:

- **Zero log-probabilities are computed anywhere** (`grep output_scores|transition_scores|logprob` → 0 hits). No length normalization, no Rao-Blackwellization. Mass = raw sample counts.
- NLI merging is gated on `answer_form ∈ {short, entity}`, so **9 of 13 committed cells (all GSM8K/MATH) use exact-string clustering only**.
- `1 − H/log(n_valid)` normalizes by *valid* samples, so a question with one parsed answer scores behavioral confidence **1.0**. Affects 7.9% of rows.

### 4. Three signals are not commensurable — the deepest design issue
One unsupervised signal (entropy) is raced against **two supervised accuracy predictors**
(Format B; the probe, which uses `PROBE_LABEL="correct"` — PLAN §6·5 recommended the
entropy label, unused). All three are then calibrated to P(correct).

Each is also measured on a *different* generation (B / SAMPLE / EXTRACT), while `correct`
comes from B and the probe trained on EXTRACT correctness — accuracy differs by up to 12
points across variants.

Not rigged toward the probe, though: unsupervised entropy **out-resolves** it, 0.079 vs 0.040.

### 5. Grid completion — 13/30 cells, none at planned size

| tier | 0.5B | 1.5B | 3B | 7B-I | 7B-base |
|---|---|---|---|---|---|
| R1 PopQA-hi | 100· | 1000 | **1000** | **1000** | 1000 |
| R2 PopQA-lo | 100· | 100· | 100· | 100· | 100· |
| R3 SimpleQA | 100· | 100· | 100· | 100· | 100· |
| C1 GSM8K | 100· | 1000 | **1000** | 1000 | 1000 |
| C2 MATH1-2 | 100· | 1000 | **1000** | **1000** | 1000 |
| C3 MATH4-5 | 100· | 100· | 100· | **1000** | 100· |

**bold** = also survives verbal pre-flight. **R2 and R3 commit zero cells. 0.5B commits
zero.** PLAN's target was N=2000; no cell reached it.

The retrieval arm of H4 is now a single tier — the *easiest* one — and the popularity
manipulation, which is the design's most distinctive feature, was never analyzed.

### 6. The quadrant result is carried by three cells
It covers 6/30 cells and 2/5 models. In 4 of those 6, `verbal_cal` never crosses 0.5
(7B|C3 range [0.134, 0.136] → all 200 rows `agree_low`). **All 224 hopeful cases come from
3 cells** whose modal bucket maps to 0.513/0.535/0.536 — 1-4 points above the cut.

H2's only passing features (`is_long`, `family`, `has_number`, `has_multi_entity`) are all
tier proxies; the one non-tier feature (`has_year`) fails.

### 7. Claims that must be withdrawn
- **H2 / "hopeful confidence"** — threshold-adjacent and cell-determined, per above.
- **H3 and Figure 3** — `hopeful_rate_base = 0.0` is a NaN coerced by `.astype(float)`
  before `.dropna()`. The "CI excluding zero" is measuring missing data.

### 8. Documentation vs data
Gate 1's `manual_correct` is **0/200 filled in all three trees**, while
`POINTS(Debojeet).md` line 6 records it "COMPLETED (97.5%)". Real strata: alias_exact
91.6%, symbolic 91.0%.

Gate 3 is computed but never enforced; H2's pass rule silently drops its "AND Gate 1
holds" conjunct; Gate 1 is `null` in the shipped run. `PROBE_C_GRID` is dead code (C=1.0
hardcoded; `auroc_train = 1.000` in 51/65 rows). `code_sha: "nogit"` — no run is bound to
a code version. PLAN §0 still says "no runs yet"; §17.2's run log is empty after three runs.

### 9. The 25-80% band: right idea, wrong implementation
Conditioning cell inclusion on accuracy **is** a legitimate base-rate control — and the
literature auditor is right that it is this project's best claim to novelty, because every
published cross-signal comparison confounds method signal with item difficulty.

But *deleting cells* is the wrong way to implement it. It removed all of R2, all of R3,
all of 0.5B, and 5/6 of C3, and selection ran on a noisy n=100 pilot (7B-Instruct C1
moved .74 → .58 between runs).

**Fix:** match on accuracy *within* cell, or control base rate statistically via the
Murphy decomposition terms. Keep the control; drop the deletion.

---

## The result nobody claimed

**Correctness is decodable at ~25% depth for both model families, with retrieval ceilings
0.84-0.91 vs reasoning 0.51-0.80.** This falsifies H4, is robust, and is the best finding
in the repo. It appears in a personal notes file, not the README.

---

## Novelty

The stated headline — verbal confidence runs hot while internals know better — was
published four separate times in 2026. The quadrant taxonomy has a direct structural twin:
[DECK: A Consistency × Confidence Taxonomy of LLM Hallucinations](https://arxiv.org/abs/2606.02289)
(Jun 2026), a 2×2 of inter-sample consistency × confidence with four named regimes across
3 models × 4 datasets. *(Verified.)* Mahaut et al. (ACL 2024) already compared verbalized /
sequence-probability / P(True) / hidden-state probes on shared items.

**What survives: the base-rate control.** That is the reframing.

---

## Improvements, prioritized

### Priority 1
1. **Fix the activation tap** — read the last *prompt* token. Every probe number needs regenerating.
2. **Withdraw H2 and H3/Figure 3.**
3. **Replace Format B as the canonical verbal axis** with a raw model-stated confidence
   that isn't label-derived. Report B separately as the supervised reference.
4. **Add log-probabilities to semantic entropy** (length-normalized, Rao-Blackwellized),
   and enable NLI merging for math answer forms.

### Priority 2
5. **Promote the depth-of-decodability result to the headline** — it is the real finding.
6. **Re-implement the base-rate control by matching, not deletion.** Recover R2/R3 — the
   popularity gradient is the paper's distinctive asset.
7. Fill Gate 1 or remove the claim that it passed; enforce Gate 3; restore H2's dropped conjunct.
8. Record `code_sha` per run; update PLAN §0 and §17.2.

### Priority 3
9. Reach N=1000 on the remaining cells; 0.5B or drop the size axis and say so.
10. Mixed-effects model over questions-in-tiers-in-models rather than independent cells.

---

## Reframing

> **"Three Measures of Confidence Disagree: Format Sensitivity, Sampling Entropy, and
> Hidden-State Decodability in Small LLMs"**

Retire "hopeful confidence" — it anthropomorphizes and will draw fire at an NLP venue.
Use "unwarranted stated confidence." Position as a **measurement-validity / negative-results
paper with a released pipeline**, which makes DECK and the 2026 systematic evaluations
targets of a critique rather than competitors.

---

## Venue ladder

| Tier | Venue | Deadline | Notes |
|---|---|---|---|
| **Primary** | **ARR 12 Oct 2026 → NAACL 2027 Findings** | 12 Oct 2026 | ~7 weeks; enough for the 7.1 GPU-hr grid plus writing. The realistic target. |
| **Secondary** | **TMLR** | rolling (~9wk) | Highest expected value — the only venue where "three replicated, one flipped sign, one untestable" is publishable rather than desk-rejected. TACL's 9-month rule forces a choice before 12 Oct. |
| **Tertiary** | **NeurIPS 2026 workshops** | ~29 Aug 2026 | 4-page flag-plant on the methodological claim from pilot data. |
| | COLM 2027 (~31 Mar) | | Main-track shot if the depth-of-decodability result holds up. |

Already missed: UncertaiNLP @ EMNLP 2026 (7 Aug), EACL 2027 (ARR 3 Aug). **Do not attempt
ICLR 2027** (24 Sep) — the grid will not be ready.

---

## Product / deployment assessment

**Verdict: the pipeline is the product, not the finding.**

| Component | Viability |
|---|---|
| Three-signal measurement pipeline | ✓ **Genuinely reusable.** Multi-model, multi-tier, molab/Kaggle portable, with honest compute accounting. |
| "Hopeful confidence" detector | ✗ Withdrawn — threshold-adjacent, cell-determined. |
| Depth-of-decodability probe | ✓ **Deployable after the tap fix.** ~25%-depth correctness probes at 0.84-0.91 AUROC on retrieval is a real abstention signal. |
| Semantic entropy scorer | ~ Works, but shortcut; fix log-probs before shipping. |

**Realistic path:** release the pipeline as an open evaluation harness — a
difficulty-controlled UQ benchmark is a real gap, and the base-rate control is exactly the
contribution the field is missing. The abstention-gating use case (probe at 25% depth,
route low-confidence queries to retrieval or a larger model) is commercially real and
cheap at one forward pass, but it is gated on fixing the activation tap first.