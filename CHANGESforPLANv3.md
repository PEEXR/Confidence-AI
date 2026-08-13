# Changes for PLAN v3

> Consolidates decisions from the post-v2 design conversation (dataset ladder,
> Qwen2.5 model ladder, Kaggle hardware, percentile-depth probing, analysis
> rework). Written to be merged into PLAN.md the same way CHANGES.md v1→v2
> was. Each item below names the PLAN.md section it replaces or amends.

---

## 1. Dataset (§3) — replaced

Single-dataset design (PopQA / SimpleQA / AA-Omniscience, pick one) is
replaced by a fixed six-tier ladder. All tiers are free-response with a short
canonical answer, so one grader family (§7) covers the whole ladder — no
format confound between tiers.

| Tier | Source | Type | Difficulty from | Answer form |
|---|---|---|---|---|
| R1 | PopQA, top popularity quintile | Retrieval | Popularity score | Short entity |
| R2 | PopQA, bottom quintile | Retrieval | Popularity score | Short entity |
| R3 | SimpleQA | Retrieval, adversarial | Dataset design | Short |
| C1 | GSM8K | Reasoning | — | Numeric |
| C2 | MATH, levels 1–2 | Reasoning | Built-in level | Short LaTeX |
| C3 | MATH, levels 4–5 | Reasoning | Built-in level | Short LaTeX |

**Dropped:** AA-Omniscience as the primary dataset (superseded by the ladder
structure — its domain tags are no longer needed as the stratification
variable). ChemBench as a scored ladder rung (organic-chem subset n=429 too
small to split reliably; 2,544 MCQ vs. 244 open-ended mixes answer format
with difficulty).

**Retained, demoted:** ChemBench stays available as an optional held-out
generalization check on the largest model only — not part of the scored
grid.

**Rationale:** the ladder is what makes H4 (retrieval vs. reasoning) testable
as a controlled gradient rather than an incidental dataset choice — difficulty
comes from within-dataset gradients (PopQA popularity, MATH level) so domain,
format, and grader stay held constant across the retrieval→reasoning move.

---

## 2. Models (§9) — replaced

Qwen3.5-0.8B baseline + LFM2.5-8B-A1B comparison is replaced by the
**Qwen2.5 same-family ladder**, plus a same-family H3 comparison pair.

**Main scaling ladder** (drives H4 — the retrieval-vs-reasoning /
depth-of-computation claim):

| Model | Layers | Hidden dim | VRAM (FP16) |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 24 | 896 | ~1 GB |
| Qwen2.5-1.5B-Instruct | 28 | 1536 | ~3 GB |
| Qwen2.5-3B-Instruct | 36 | 2048 | ~6 GB |
| Qwen2.5-7B-Instruct | 28 | 3584 | ~14 GB |

**H3 comparison pair** (base vs. post-training, at fixed scale):

- Qwen2.5-7B (base) vs. Qwen2.5-7B-Instruct (already the top rung of the
  ladder above — no extra model needed on that side)

**Dropped:** LFM2.5-8B-A1B entirely. It had a genuine VRAM bug in v2 (§10
claimed MoE gives comparable headroom to the dense baseline — wrong; MoE
reduces compute, not memory, so all ~8B expert weights must be resident,
~16GB at FP16) and, independent of that, confounded H3 against lab/data/
architecture differences rather than isolating post-training.

**Rationale for base-vs-Instruct over LFM:** same weights, same pretraining
data, differs by post-training only — a method-matched contrast instead of a
confound. Directly answers "does instruction-tuning change the hopeful/
suppressed confidence mismatch rate" without the MoE/lab confound.

**Flag for implementation:** the ladder above assumes Instruct variants
throughout, since verbalized-confidence elicitation (§4, formats A/B/C)
needs a model that reliably follows the elicitation prompt — base models
are a poor fit for that. Qwen2.5-7B-base is added *only* as the H3
comparison point, not as a ladder rung. Confirm this split makes sense
before running E1 (format agreement) on the base model.

---

## 3. Sample size per cell — resolved

**Was:** 300–500 questions (compute-constrained assumption from the RX 6600
plan).

**Now:** larger target, ~2000 questions per cell, per the Kaggle-compute
discussion (2000 questions × N=10 sampling is minutes at 1.5B, one to two
hours at 7B; a full grid lands in the 10–15 GPU-hour range on a 4-model
ladder, which fits inside the weekly quota with room for reruns).

**Still applies unchanged from §8's difficulty-definition rule:** pilot 100
questions per cell first, keep only cells landing 25–80% accuracy before
committing to the full 2000. The larger target is a ceiling for cells that
clear the pilot, not a blind per-cell default — a ragged grid (some cells
excluded) is still an acceptable, reportable outcome.

**Open follow-up:** with 5 model variants (4-rung ladder + 7B-base) × 6
tiers = 30 cells, re-check the total GPU-hour estimate against the 30
hr/week Kaggle cap before committing all 30 cells to the full 2000-question
target — the original 10–15 hr estimate was sized for a 4-model × 4-tier
grid, not 5×6.

---

## 4. Internal confidence extraction (§6) — methodology clarified

Still a single greedy forward pass (temp=0), still extracted at the **last
prompt token position only** — no during-generation / multi-timestep
probing (see rationale below). What changes: extraction now sweeps **five
fixed depth percentiles** per question at that one position, instead of
selecting one layer directly.

| Model | 0% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| 0.5B (24 layers) | 0 | 6 | 12 | 18 | 24 |
| 1.5B (28 layers) | 0 | 7 | 14 | 21 | 28 |
| 3B (36 layers) | 0 | 9 | 18 | 27 | 36 |
| 7B (28 layers) | 0 | 7 | 14 | 21 | 28 |

Train one logistic regression per (question set × percentile), same
position, different layer — this isolates depth as the sole swept variable.
The single "winning layer" selection (by calibration-split AUROC, §6 step 6)
is unchanged and still the headline number per cell; the full 5-point sweep
is additionally reported as a **depth curve** — AUROC vs. percentile, one
line per tier, faceted by model size. That figure is the direct visual test
of H4.

**Explicitly not doing:** probing at multiple token positions during
generation (i.e., after 25%/50%/75% of the generated answer). That is a
different, currently crowded area of the literature (multiple 2025–2026
papers already probe hidden states across CoT generation steps, including
scale sweeps on reasoning benchmarks). Keeping extraction at the
last-prompt-token position and sweeping only depth keeps this project in a
comparatively open lane: pre-generation depth-of-signal, contrasted across
task type (retrieval vs. reasoning) and model scale, together — that
specific combination wasn't found in the pass done during design.

---

## 5. Hardware (§10) — rewritten for Kaggle

RX 6600 8GB, ROCm assumptions are replaced with the Kaggle GPU runtime.

- **2× NVIDIA T4, 16GB each, no NVLink** (not a pooled 30GB — per-device
  ceiling is 16GB). Use both as independent workers (split questions across
  devices) rather than sharding — nothing in the current model ladder needs
  sharding, since 7B fits a single T4.
- **~30 GPU-hours/week, ~12-hour session cap.** Every long job must
  checkpoint and resume by question ID.
- **/kaggle/working is the only persistent output space.** Cache model
  weights as a Kaggle Dataset (mount read-only) to avoid re-pulling ~15GB
  from HF every session.
- **P100 fallback:** hardware allocation isn't guaranteed; write
  device-detection into the loading code rather than hardcoding two T4s.
- **T4 is Turing: no native BF16.** Qwen2.5 ships in BF16; running FP16-only
  risks occasional inf/NaN in later-layer activations. See Gate 3 update
  below — this is now an explicit checked risk, not an assumed non-issue.

---

## 6. Extraction mechanics (§6/§10) — clarified, not changed in spirit

`output_hidden_states=True` is out — it materializes every layer × every
token position (tens of GB at this question count, blows the output cap).
Replaced by forward hooks that grab only the five percentile-layer vectors
at the last prompt token. Storage impact of going from 1 layer to 5 per
question: ~40KB/question instead of ~8KB — still trivial against the 20GB
cap even at 2000 questions × 6 tiers × 5 model variants.

---

## 7. Analysis (§8) — headline metric changed

Raw per-cell ECE/Brier as the headline is dropped — with 30 cells at
varying base rates, "calibration improved" would be inseparable from
"accuracy improved." Replaced by:

1. **Murphy decomposition** of Brier into reliability (calibration),
   resolution (discrimination), and uncertainty (base rate). Report
   reliability and resolution separately per signal instead of raw Brier.
2. **One hierarchical logistic regression pooled across the whole grid**,
   with question-level random effects, instead of 30 independent per-cell
   estimates:
   ```
   correct ~ verbal + behavioral + internal(best_layer) +
             tier + log(params) +
             tier:internal +            # does probe validity drop on reasoning?
             log(params):internal +     # does probe validity rise with scale?
             layer_pct:tier             # does signal-onset depth shift by task type?
   ```
3. Difficulty still defined per-model via the 100-question pilot (§3 change
   above), keeping cells in the 25–80% accuracy band.
4. Verbal signal pre-flight check unchanged: exclude a cell from the verbal
   comparison if it produces fewer than ~3 distinct confidence values,
   rather than reporting it as "poorly calibrated."

---

## 8. Gate 3 — probe validity (§16) — pass rule updated

**Was:** winning layer beats label-shuffle null and surface baseline, train
AUROC ≥ 0.65.

**Now:** for each (tier × model) cell, the probe across the full 5-percentile
sweep beats the label-shuffle null and surface baseline on the calibration
split, with at least one percentile reaching AUROC ≥ 0.65.

**Added check, ahead of trusting any AUROC from this gate:** assert every
extracted activation tensor is finite; log the fraction of non-finite values
per layer. T4's FP16-only path is a real risk for silent corruption in late
layers — this turns "the probe doesn't work" from an ambiguous result into a
diagnosable one (dirty activations vs. genuinely no signal).

---

## 9. H3 — resolved, not dropped

**Decision:** H3 stays in the plan. Model pair changes from
Qwen3.5-0.8B-vs-LFM2.5-8B-A1B to **Qwen2.5-7B (base) vs. Qwen2.5-7B-Instruct**
— see §2 above for the model table and rationale. Hypothesis statement,
predicted direction, and falsification condition in §13 H3 carry over
unchanged; only the model identities change. Gate 4's pass rule (delta CI
excludes 0, missed-knowledge guard doesn't rise to offset it) is unaffected.

---

## 10. Carried over unchanged

- §4 Verbalized confidence: three-format elicitation (numeric / bucket /
  forced-action), empirical bucket→probability mapping, format-agreement
  check (H0 / Gate 2)
- §4.1 Abstention granularity: justified-hedge vs. missed-knowledge split
  via forced-answer companion
- §5 Semantic entropy methodology (N=10, T=0.7–1.0, clustering, Shannon
  entropy)
- §7 Ground truth labeling (normalized string match → NLI/LLM-judge
  escalation)
- §8.2 Omniscience-Index as companion metric
- §14 Controls (label-shuffle null, layer sweep on calibration split only,
  bag-of-words baseline, etc.)
- H1, H2 as originally stated
- No fine-tuning anywhere in the pipeline (§10 note) — still true; the base
  models stay frozen throughout, only the auxiliary logistic regressions and
  isotonic/Platt calibration functions are fit

---

## 11. Still open

1. **§15 predicted-results figures** haven't been redrawn for the
   depth-curve design. Needs a new predicted-vs-null sketch: depth curves
   for retrieval tiers (flat, high AUROC from 0%) vs. reasoning tiers
   (chance until late layers) — this is the H4 figure and doesn't exist yet
   in pre-registered form.
2. **30-cell compute budget** (§3 above) — needs a sanity check against the
   30 GPU-hr/week Kaggle cap before locking in the full 2000-question target
   for every cell.
3. **Base-model elicitation flag** (§2 above) — confirm Qwen2.5-7B-base can
   produce usable output for the verbalized-confidence formats (§4) before
   assuming E1/E2 port over unchanged to the H3 comparison; base models are
   generally worse at instruction-following, so this may need a modified
   prompt or a documented limitation.
