# Genuine vs. Hopeful Confidence in LLMs — Research Plan (v3)

> **Revision note (Aug 2026):** This is the final revision. It merges
> CHANGESforPLANv3.md into v2: the dataset is now a fixed six-tier
> retrieval→reasoning ladder (§3); the model ladder is the Qwen2.5 same-family
> Instruct series plus a 7B base-vs-Instruct H3 pair (§9); hardware moved to
> Kaggle's 2× T4 runtime (§10); internal-confidence extraction sweeps five
> fixed depth percentiles via forward hooks (§6); and the analysis headline
> became a Murphy decomposition plus one hierarchical logistic regression
> pooled across the grid (§8). H4 is added as a new hypothesis (§13).

## 0. Status

**Status: pre-registration — no runs yet.**

**Open items carried into v3 (each is tracked as a standing risk, §16):**
1. The §15 depth-curve figure stub (`plots/fig4_depth_prediction.py`) has not
   been written yet — it must be pre-registered before any run (§17.3).
2. The 30-cell compute budget (5 model variants × 6 tiers) needs a sanity
   check against the 30 GPU-hr/week Kaggle cap before all cells commit to the
   2000-question target (§10).
3. Qwen2.5-7B-base must be confirmed to yield usable output for the
   verbalized-confidence formats before E1/E2 assumptions port over to the H3
   comparison (§9 flag).

## 1. Research question

LLM "confidence" isn't one thing. This project measures it three independent
ways on the same question set and studies where they **disagree**:

1. **Verbalized confidence** — what the model *says* about its own certainty
2. **Behavioral confidence** — how consistent the model's answers are when
   sampled repeatedly (semantic entropy)
3. **Internal confidence** — what a linear probe on hidden states predicts,
   from a single forward pass

The core finding to look for: cases of **"hopeful" / performed confidence**
(model states high confidence, but internals/behavior say it's guessing) vs.
**genuine confidence** (all three agree) vs. **suppressed confidence**
(model hedges in words despite internals/behavior showing real knowledge).

Orthogonal axis this plan fixes: **task type**. The question set is a fixed
six-tier retrieval→reasoning ladder (§3), so the *depth at which internal
confidence separates from chance* can be contrasted across task type and
model scale as its own finding (H4, §13) rather than being an incidental
artifact of dataset choice.

Headline metrics: per-signal reliability and resolution from the Murphy
decomposition of Brier (§8); per-signal ECE/Brier on the test split for the
three-signal calibration comparison (H1); how much the three disagree with
each other (Spearman correlation + quadrant analysis of mismatches); and the
depth-of-signal contrast across task type and scale (H4).

> Pre-registration apparatus: hypotheses [§13](#hypotheses), controls
> [§14](#controls), predicted results [§15](#predicted-results), checkpoints
> [§16](#checkpoints), experiments [§17](#experiments). Status tracked in
> [§0](#status).

## 2. Motivation / context

- Thinking Machines' **Inkling** (July 15, 2026) explicitly trains for
  "epistemics" — calibration via RL against proper scoring rules on resolved
  real-world questions, plus dual rubric/claims graders. This shows the
  industry treats calibration as trainable, but doesn't tell us whether
  verbalized confidence tracks the model's *actual* internal state.
- The filler/pause-token literature (Goyal et al. 2023 "Think Before You
  Speak"; Pfau et al. 2024 "Let's Think Dot by Dot"; recent frontier-scale
  work like "Reading Between the Dots") shows models can do real hidden
  computation over content-free tokens. Related, but orthogonal to this
  project's main question — see Stretch Goals.
- Prior art this project builds on / must not re-derive from scratch:
  - Kadavath et al. 2022, *Language Models (Mostly) Know What They Know*
  - Kuhn et al. 2023 / Farquhar et al. 2024 (*Nature*), semantic entropy
  - **Kossen et al. 2024, Semantic Entropy Probes** — closest prior work;
    this project is essentially an extended, three-way version of their setup
  - Orgad et al. 2024, *LLMs Know More Than They Show*
  - Azaria & Mitchell 2023, SAPLMA; Burns et al. 2022, Discovering Latent
    Knowledge
  - Lindsey (Anthropic) 2026, *Emergent Introspective Awareness in LLMs* —
    most relevant framing for "genuine vs. confabulated self-report"

**Scope honesty:** LLM calibration is a crowded field. The contribution here
is not "confidence is a new topic," it's a clean three-signal comparison on a
small same-family model ladder (0.5B–7B) with characterization of the
disagreement cases, plus a controlled retrieval-vs-reasoning depth-of-signal
contrast (H4).

## 3. Dataset

A fixed **six-tier ladder** — no single-dataset design. All tiers are
free-response with a short canonical answer, so one grader family (§7) covers
the whole ladder; there is no format confound between tiers.

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
generalization check on the largest model only — not part of the scored grid.

**Rationale:** the ladder is what makes H4 (retrieval vs. reasoning) testable
as a controlled gradient rather than an incidental dataset choice — difficulty
comes from within-dataset gradients (PopQA popularity, MATH level), so domain,
format, and grader stay held constant across the retrieval→reasoning move.

**Per-cell pipeline:**
- **Pilot 100 questions per cell first.** Keep only cells landing in the
  25–80% accuracy band before committing to the full target (the §8
  difficulty-definition rule).
- **Full target: ~2000 questions per cell** for cells that clear the pilot —
  a ceiling, not a blind per-cell default. A ragged grid (some cells
  excluded) is an acceptable, reportable outcome.
- Split inside each cell: **60% train / 20% calibration / 20% test**.
- **Sanity check before running the full pipeline:** hand-verify correctness
  on 50–100 questions per grader family. If automated grading disagrees with
  manual check on more than ~5%, fix grading before trusting downstream
  numbers (Gate 1, §16).

## 4. Signal 1 — Verbalized confidence

Elicit in **three separate formats**, check they agree, before treating any
one as "verbalized confidence":

| Format | Prompt style | Known failure mode |
|---|---|---|
| A. Numeric | "State confidence 0–100%" | Clumps at round numbers (50/70/90/95/100) |
| B. Verbal bucket | Certain / Fairly confident / Somewhat unsure / Mostly guessing / No idea | Needs empirical calibration of bucket → probability, not hand-assigned values |
| C. Forced action / betting | Answer or Pass, with asymmetric payoff for wrong vs. right | Closest to how Inkling itself is trained (proper scoring rules) |

**Bucket-to-probability mapping (format B):** use the *actual* ground-truth
accuracy of all answers the model placed in a given bucket — not a
hand-assigned number.

**Agreement check:**
1. Run all three formats on the same 50–100 question subset, temp=0, single
   sample.
2. Convert to a common 0–1 scale.
3. Spearman correlation pairwise across formats.
4. ECE/Brier per format against ground truth — pick the best-calibrated
   format as canonical "verbalized confidence" going forward.
5. If formats correlate well (Spearman > ~0.6–0.7): collapse to the
   best-calibrated one. If they don't: that disagreement is itself a
   reportable finding — report all three separately rather than picking one.

**Grading correctness:** normalized string match (lowercase, strip
punctuation/articles) as the default; escalate to NLI entailment
(`microsoft/deberta-large-mnli`) or an LLM-judge call for messier/open-ended
answers.

### 4.1 Abstention granularity — justified hedge vs. missed knowledge

A "Pass" under Format C is currently just one bucket. But a Pass could mean
two very different things: the model correctly sensed it didn't know (good
abstention), or the model actually knew the answer but hedged anyway
(suppressed confidence — same phenomenon as the low-verbal/high-internal
quadrant in Section 8, just showing up via the betting frame instead).

1. For every question where the model chose **Pass** under Format C, run a
   **second, forced-answer pass**: same question, same context, but the
   Pass option removed — model must produce an answer.
2. Grade that forced answer against ground truth as usual.
3. Split every Pass into:
   - **Justified hedge** — Passed, and the forced answer would have been
     wrong. Abstention was the right call.
   - **Missed knowledge** — Passed, but the forced answer would have been
     right. The model knew it, but hedged — folds into the "suppressed /
     hopeful-in-reverse" side of the quadrant analysis in Section 8.
4. Report the split as counts/percentages alongside the existing quadrant
   breakdown, and pull example questions from the "missed knowledge" bucket
   the same way Section 8 already pulls examples for the other quadrants.

**Caveat to state explicitly in the writeup:** a forced answer is not
necessarily identical to what the model "would have said" if answering had
stayed optional — being compelled to answer can itself change the output.
Report this as "accuracy under compulsion," not as ground truth about the
original hedge.

## 5. Signal 2 — Behavioral confidence (semantic entropy)

1. Sample **N=10** generations per question, **T=0.7–1.0** (must have real
   sampling variance — T=0 gives entropy of 0 always).
2. **Cluster answers by meaning:**
   - Fast path: string normalization (lowercase, strip punctuation, extract
     canonical form e.g. year/entity) — covers most short factual answers.
   - Robust path: bidirectional NLI entailment between answer pairs, feed
     into agglomerative clustering (`scipy.cluster.hierarchy`).
3. **Entropy over cluster sizes** (clusters as probability mass):
   ```python
   p = cluster_sizes / n_samples
   H = scipy.stats.entropy(p)          # Shannon entropy
   H_max = np.log(n_samples)
   confidence_behavioral = 1 - (H / H_max)
   ```
4. Sanity check: all-same-answer → H=0 → confidence=1. Evenly split answers
   → H≈H_max → confidence≈0.

## 6. Signal 3 — Internal confidence (linear probe)

1. **Single greedy forward pass** per question (temperature=0). Deliberately
   cheap — one pass, not ten — since the point is testing whether internals
   recover the behavioral signal without the sampling cost.
2. **Extract hidden state at the last prompt token position** only — no
   during-generation / multi-timestep probing (see note below). Cleaner than
   last-generated-token, no look-ahead bias from the model's own answer.
3. **Sweep five fixed depth percentiles** per question at that one position —
   same position, different layer, isolating depth as the sole swept
   variable:

   | Model | 0% | 25% | 50% | 75% | 100% |
   |---|---|---|---|---|---|
   | 0.5B (24 layers) | 0 | 6 | 12 | 18 | 24 |
   | 1.5B (28 layers) | 0 | 7 | 14 | 21 | 28 |
   | 3B (36 layers) | 0 | 9 | 18 | 27 | 36 |
   | 7B (28 layers) | 0 | 7 | 14 | 21 | 28 |

4. **Train one logistic regression per (question set × percentile)**, with
   `StandardScaler` first. Use a 1-layer MLP or PCA pre-reduction only if
   logistic regression clearly underfits (train AUROC < ~0.65).
5. **Label options:**
   - Binary: ground-truth correctness
   - Continuous: the semantic entropy value from Signal 2 (this is the
     actual Semantic Entropy Probes approach — more interesting since it
     directly tests "does a single forward pass recover the 10-sample
     behavioral signal?")
6. **Select best layer** by AUROC on the **calibration** split (never on
   train). The single winning layer remains the headline number per cell.
7. **Report the full 5-point sweep as a depth curve** — AUROC vs. layer
   percentile, one line per tier, faceted by model size. That figure is the
   direct visual test of H4 (§13).
8. **Calibrate** the winning probe's raw output via isotonic regression
   fit on the calibration split, applied to the test split.

**Extraction mechanics:** `output_hidden_states=True` is out — it
materializes every layer × every token position (tens of GB at this question
count, blows the output cap). Replaced by forward hooks that grab only the
five percentile-layer vectors at the last prompt token. Storage going from
1 layer to 5 per question: ~40KB/question instead of ~8KB — still trivial
against the cap even at 2000 questions × 6 tiers × 5 model variants.

**Explicitly not doing:** probing at multiple token positions during
generation (i.e., after 25%/50%/75% of the generated answer). That is a
different, currently crowded area of the literature (multiple 2025–2026
papers already probe hidden states across CoT generation steps, including
scale sweeps on reasoning benchmarks). Keeping extraction at the
last-prompt-token position and sweeping only depth keeps this project in a
comparatively open lane: pre-generation depth-of-signal, contrasted across
task type (retrieval vs. reasoning) and model scale, together.

**Note on "training":** the base model itself is never fine-tuned anywhere
in this pipeline (see Section 10). The logistic regression / MLP here is a
small auxiliary classifier fit on top of frozen hidden states — the model
weights never receive gradients. Same applies to the isotonic/Platt
calibration functions in Section 8.

## 7. Ground truth

- Binary correct/incorrect label per question, independent of all three
  confidence signals above.
- Default: normalized string match against reference answer.
- Escalate to NLI entailment or LLM-judge for paraphrase-heavy or
  open-ended answers.
- Report the labeling method and manual-agreement rate in the writeup.

## 8. Cross-signal comparison

**Problem:** raw scores from the three signals are all in [0,1] but don't
necessarily mean the same thing (verbalized confidence is known to run hot
relative to true accuracy). Range-normalizing isn't enough — need to
calibrate each signal against ground truth independently before comparing
magnitudes.

1. Fit a **per-signal calibration function** (isotonic regression, or Platt
   scaling if the calibration split is small / isotonic overfits) mapping
   raw score → empirical P(correct), fit on the calibration split only.
2. Apply each function to the **test** split. Now `calibrated_verbal`,
   `calibrated_behavioral`, `calibrated_internal` are genuine P(correct)
   estimates on the same held-out questions.
3. **Headline metric — Murphy decomposition, not raw per-cell ECE/Brier.**
   With 30 cells at varying base rates, "calibration improved" would be
   inseparable from "accuracy improved." Instead decompose Brier into three
   additive components and report them separately per signal:
   - **Reliability** (calibration component)
   - **Resolution** (discrimination component)
   - **Uncertainty** (base-rate component, reported for context)
   Report reliability and resolution per signal instead of raw Brier.
4. **One hierarchical logistic regression pooled across the whole grid** (not
   30 independent per-cell estimates), with question-level random effects:
   ```
   correct ~ verbal + behavioral + internal(best_layer) +
             tier + log(params) +
             tier:internal +            # does probe validity drop on reasoning?
             log(params):internal +     # does probe validity rise with scale?
             layer_pct:tier             # does signal-onset depth shift by task type?
   ```
5. **Difficulty control:** difficulty is defined per-model via the 100-question
   pilot (§3), keeping cells in the 25–80% accuracy band.
6. **Verbal signal pre-flight:** exclude a cell from the verbal comparison if
   it produces fewer than ~3 distinct confidence values — exclusion, not
   reporting it as "poorly calibrated."
7. **Other metrics to report:**
   - Pairwise Spearman correlation between the three raw signals (rank-order
     only, no calibration required) — raw; Pearson on calibrated.
   - **Omniscience-Index** (decision-level companion metric — see 8.2)
8. **Quadrant / mismatch analysis** — pull actual examples:
   - High verbal + low behavioral/internal → **"hopeful" / performed
     confidence** (likely headline finding)
   - Low verbal + high behavioral/internal → suppressed confidence /
     hedging despite real knowledge (possible safety-training artifact)
   - **Missed knowledge** cases from Section 4.1 fold into this same
     low-verbal/high-internal quadrant via the Format C betting frame —
     report alongside the verbal-hedging cases, not as a separate category
   - Characterize what kinds of questions land in each quadrant (long-tail
     entities, dates, ambiguous phrasing, etc.)
9. **Per-model comparison:** run 1–8 above for each model separately, then
   report deltas — specifically, does the instructed model (7B-Instruct)
   show a lower rate of high-verbal/low-behavioral-or-internal ("hopeful
   confidence") cases than its base counterpart (7B-base)? Report per-cell
   decomposition side by side; the delta between them is the actual finding,
   not just two separate result tables.

**If the dataset (any cell) is too small for calibration to be trustworthy:**
drop magnitude comparison for that cell, keep Spearman correlation + raw-score
quadrant bucketing, and state this as an explicit scope limitation in the
writeup — that's a legitimate call, not a flaw.

### 8.2 Omniscience-Index

ECE and Brier measure whether stated probabilities are honest, but neither
directly scores the decision-level question "did the model correctly choose
to answer or abstain?" This is a simple, cheap addition on top of data
already collected:

```
Index = (n_correct − n_incorrect) / n_total × 100
```

- +1 for correct, −1 for incorrect, 0 for abstained (Format C Pass) — no
  penalty for abstaining itself, since there's no observable counterfactual
  for an unanswered question.
- Report per model (base vs Instruct, Section 9) and per tier.
- Companion summary statistic, not a replacement for reliability/resolution —
  keep both.

**Why abstain = 0 and not split into +1/−1:** a benchmark score can only
grade what was actually observed. An abstained question has no produced
answer, so there's no ground-truth comparison to make — scoring it would
require the forced-answer pass from Section 4.1. Where that data exists (via
the Format C forced-answer companion), use the finer-grained justified/missed
split from 4.1 for the qualitative analysis; keep the Index itself at the
standard +1/−1/0 scoring so it stays comparable to published AA-Omniscience
numbers for other models.

## 9. Models

**Main scaling ladder** — Qwen2.5 same-family Instruct models, each run
through the full pipeline (Sections 4–8) independently:

| Model | Layers | Hidden dim | VRAM (FP16) |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 24 | 896 | ~1 GB |
| Qwen2.5-1.5B-Instruct | 28 | 1536 | ~3 GB |
| Qwen2.5-3B-Instruct | 36 | 2048 | ~6 GB |
| Qwen2.5-7B-Instruct | 28 | 3584 | ~14 GB |

Instruct variants throughout, since verbalized-confidence elicitation (§4,
formats A/B/C) needs a model that reliably follows the elicitation prompt —
base models are a poor fit for that.

**H3 comparison pair** (fixed scale, post-training contrast):

- **Qwen2.5-7B (base)** vs. **Qwen2.5-7B-Instruct** (already the top rung of
  the ladder above — no extra model needed on that side). The base model is
  added *only* as the H3 comparison point, not as a ladder rung.

**Dropped:** LFM2.5-8B-A1B entirely. It had a genuine VRAM bug in v2 (MoE
reduces compute, not memory — all ~8B expert weights must be resident, ~16GB
at FP16) and, independent of that, confounded H3 against lab/data/
architecture differences rather than isolating post-training.

### 9.1 Why same-family scaling + full precision instead of large/quantized

1. **Same-family scaling keeps architecture held constant** across the scale
   axis — depth, width, head count change; nothing else. Any depth-of-signal
   differences across size are attributable to scale itself, not to
   architecture-family changes (this is the H4 scale axis).
2. **Full precision = clean activations.** Signal 3's linear probes train
   directly on hidden states; quantization noise corrupts that feature space
   in ways that are hard to distinguish from genuine calibration signal.
3. **Base-vs-Instruct is a method-matched contrast:** same weights, same
   pretraining data, differs by post-training only. Directly answers "does
   instruction-tuning change the hopeful/suppressed confidence mismatch
   rate" without the MoE/lab confound.
4. Smaller models tend to show clearer overconfidence effects, which likely
   makes the quadrant mismatches in Section 8 more visible rather than
   weaker.

**Flag for implementation — base-model elicitation:** confirm before running
E1 (format agreement) that Qwen2.5-7B-base can produce usable output for the
verbalized-confidence formats (§4). Base models are generally worse at
instruction following, so this may need a modified prompt or a documented
limitation. Standing risk (§16).

## 10. Hardware / compute notes — Kaggle

- **No fine-tuning anywhere in this pipeline.** All models run in
  inference-only mode across all three signals; the only things "trained"
  are the small auxiliary models in Sections 6 and 8 (logistic regression,
  isotonic/Platt calibration) fit on extracted features on CPU — negligible
  VRAM cost, no gradients touch the base model. The base models stay frozen.
- **GPU: 2× NVIDIA T4, 16GB each, no NVLink.** Not a pooled 30GB —
  per-device ceiling is 16GB. Use both as independent workers (split
  questions across devices) rather than sharding; nothing in the current
  model ladder needs sharding, since 7B fits a single T4.
- **~30 GPU-hours/week, ~12-hour session cap.** Every long job must
  checkpoint and resume by question ID.
- **`/kaggle/working` is the only persistent output space.** Cache model
  weights as a Kaggle Dataset (mount read-only) to avoid re-pulling ~15GB
  from HF every session.
- **P100 fallback:** hardware allocation isn't guaranteed; write
  device-detection into the loading code rather than hardcoding two T4s.
- **T4 is Turing: no native BF16.** Qwen2.5 ships in BF16; running FP16-only
  risks occasional inf/NaN in later-layer activations. This is an explicit
  checked risk, not an assumed non-issue — Gate 3 runs a mandatory finiteness
  pre-check (§16).
- **Compute budget note.** Full grid = 5 model variants × 6 tiers = 30 cells.
  At ~2000 questions/cell (N=10 sampling is minutes at 1.5B, one to two hours
  at 7B), a 4-model × 4-tier grid lands in the 10–15 GPU-hour range. Re-check
  the 30-cell budget against the 30 GPU-hr/week cap before committing all
  cells to the full 2000-question target (standing risk, §16).
- The **N=10 sampling step** for semantic entropy (Section 5) is the
  expensive part regardless of model size — batch it, keep question count
  modest until the pipeline is validated end-to-end, then scale up.
- **Extraction memory:** forward hooks grab only the five percentile-layer
  vectors at the last prompt token (§6) — no `output_hidden_states=True`
  materialization; call `torch.cuda.empty_cache()` between batches.

## 11. Stretch goal

**Filler tokens × calibration.** Does giving the model dots/pause tokens
before answering change its *calibration*, not just its accuracy? This
intersection doesn't appear to be covered by existing filler-token or
calibration literature — narrower and more novel than the base three-signal
comparison, worth a follow-up section if time allows.

## 12. Deliverable / venue

- Workshop paper.
- Either outcome (signals agree vs. disagree) is a reportable result —
  don't chase a specific finding, report what the calibration curves and
  quadrant analysis actually show.

## 13. Hypotheses

> **Status:** pre-registered — none tested yet. Update after runs; claims
> that move are marked in place, never silently edited.

One card per hypothesis. Estimators and pass rules are fixed here, before any
data — an estimator chosen after the curves are seen is not an estimator.

### H0 · tier 1 · no training required

- **Statement.** The three verbalized-confidence formats (A numeric, B verbal
  bucket, C forced action, §4) elicit the same confidence on the same
  questions.
- **Predicts.** High pairwise agreement across formats on the 50–100-question
  agreement subset (temp=0, single sample).
- **Falsified if.** Any pairwise Spearman lower CI bound < 0.6.
- **Why it matters.** If formats disagree, "verbalized confidence" is not one
  thing and must be reported per format; that disagreement alone is
  publishable.
- **Estimator + pass rule.** Pairwise Spearman correlation with bootstrap CIs
  across the three formats on the agreement subset; pass if all three lower
  bounds ≥ 0.6. If passed, collapse to the best-calibrated format (lowest
  ECE, §4). H0 needs no probe, no sampling, no comparison model — it runs on
  the agreement subset and is the publishable-alone abort branch.
- **Experiment.** E1.

### H1 · tier 1 · no training required

- **Statement.** The three confidence signals are not interchangeably
  calibrated: verbalized confidence runs hot relative to behavioral and
  internal confidence.
- **Predicts.** On the test split, verbalized confidence has a worse
  calibration error (ECE/Brier) than behavioral and internal confidence by
  more than noise.
- **Falsified if.** Bootstrap CIs on per-signal ECE/Brier overlap so no
  ordering is distinguishable.
- **Why it matters.** This is the core "confidence isn't one thing" claim:
  the signals misreport different things, so a single calibration number from
  one signal is misleading.
- **Estimator + pass rule.** Per-signal ECE and Brier on the test split with
  bootstrap CIs; pass if the CI on Δ(best − worst calibrated signal)
  excludes 0.
- **Experiment.** E2.

### H2 · tier 2 · requires probe (§6) + sampling (§5)

- **Statement.** Mismatch cases are systematic, not noise: questions landing
  in the "hopeful" (high verbal, low behavioral/internal) and "suppressed"
  (low verbal, high behavioral/internal) quadrants cluster by question type,
  and "missed knowledge" cases (§4.1) land in the suppressed quadrant.
- **Predicts.** Quadrant membership is significantly associated with question
  features (domain, long-tail entities, dates, ambiguous phrasing);
  hopeful-confidence cases are not a random tail of grading noise.
- **Falsified if.** The quadrant × question-type association on held-out
  questions does not beat the label-shuffle null, or the mismatches are
  dominated by grading noise.
- **Why it matters.** Turns anecdotal "confident and wrong" cases into a
  characterisable phenomenon with question-level predictors.
- **Estimator + pass rule.** Contingency test (chi-square / log-odds) of
  quadrant membership × question features on the test split; pass if p<0.05
  AND the association survives the label-shuffle null AND Gate 1 (grading
  sanity, §16) holds.
- **Experiment.** E2, E3, E4.

### H3 · tier 3 · requires comparison model (§9)

- **Statement.** Post-training (instruction tuning) reduces "hopeful"
  confidence without trading it for blanket hedging.
- **Predicts.** Qwen2.5-7B-Instruct shows a lower rate of hopeful-confidence
  cases (high verbal, low behavioral/internal) than its base counterpart
  Qwen2.5-7B, and the reduction is not offset by a rise in "missed knowledge"
  (suppressed) cases (§4.1).
- **Falsified if.** The delta in hopeful-confidence rate between models has a
  CI including 0, or abstention is explained by blanket hedging (the
  missed-knowledge guard triggers).
- **Why it matters.** Turns pure characterisation into a comparison: is
  calibration trainable, and does the post-training do the right thing rather
  than just hedge more? Same weights and pretraining data — differs by
  post-training only (§9.1).
- **Estimator + pass rule.** Bootstrap CI on the difference of
  hopeful-confidence rates (base − Instruct) on matched question sets; pass if
  the lower bound > 0 AND the missed-knowledge rate does not rise by ≥ the
  amount hopeful confidence falls.
- **Experiment.** E6.

### H4 · tier 3 · requires full ladder + percentile sweep (§3, §6, §9)

- **Statement.** The depth at which internal confidence separates from chance
  depends on task type: for retrieval, probe AUROC beats chance by ~0% depth;
  for reasoning, probe validity onsets later (higher layer percentile) and may
  not beat chance at small scale.
- **Predicts.** On the calibration split, retrieval tiers (R1/R2/R3) reach a
  winning-layer AUROC ≥ 0.65 at a lower depth percentile, with a higher
  asymptote, than reasoning tiers (C1/C2/C3); larger models onset earlier and
  asymptote higher. The depth curve (§6 · 7) is the direct visual test.
- **Falsified if.** Depth curves for retrieval and reasoning overlap in onset
  percentile, or no percentile reaches AUROC ≥ 0.65 for reasoning at any
  scale — with cells held in the 25–80% accuracy band (§8) to control for
  accuracy.
- **Why it matters.** The retrieval→reasoning gradient is the controlled
  manipulation that makes "hopeful confidence" machine-legible: if internal
  signal lives only in late layers for reasoning (or nowhere at small scale),
  then verbal confidence on reasoning questions is genuinely harder to ground
  in internals — a mechanistic explanation for H1's verbal-hot effect.
- **Estimator + pass rule.** Per (tier × model) AUROC-vs-percentile curve on
  the calibration split against the label-shuffle null; compare onset
  percentile (first percentile reaching AUROC ≥ 0.65) across tiers with a
  bootstrap CI on the onset difference; pass if the retrieval−reasoning onset
  CI excludes 0 and Gate 3 holds for affected cells.
- **Experiment.** E2 (ladder runs), E4 (sweep).

## 14. Controls

Consolidated from §4–§8 so the checks are auditable in one place. Grouped as
measurement nulls, robustness & construct, manipulation checks, and reporting
discipline.

### 14.1 Measurement nulls

- **Label-shuffle null** for the probe (§6): the winning layer's AUROC must
  beat a probe trained on shuffled labels.
- **Bag-of-words / lexical baseline** for any valence-like or coherence claim
  (§4, §8): does a cheap surface model reproduce the effect?
- **Prompt-only classifier** for any self-report / privileged-access claim
  (§4 verbalized): must beat a classifier given the prompt only.
- **Random-choice baseline** for any coherence claim: random answers must not
  reproduce the effect.

### 14.2 Robustness & construct

- **Percentile sweep on the calibration split only** (§6): the winning layer
  is selected on calibration, never on train or test.
- **Activation finiteness** (§6/§10): every extracted activation tensor must
  be finite; the fraction of non-finite values is logged per layer (Gate 3
  pre-check).
- **Test–retest at fixed temperature** (§5): N=10 sampling at T≥0.7 must show
  real variance; a degenerate run (T=0) is excluded.
- **Probe beats a surface/embedding baseline** (§6) before its readout means
  anything.
- **Report both raw and calibrated scores** (§8): Spearman on raw, Pearson on
  calibrated.

### 14.3 Manipulation checks

- **Bucket → probability mapping is empirical** (§4, format B): actual
  per-bucket accuracy, never hand-assigned values.
- **Grading clears the ~5% manual-agreement bar** (§3, Gate 1 §16).
- **Format-agreement check** (§4) — the load-bearing positive control: the
  three verbalized formats must agree (H0) before any one is treated as
  "verbalized confidence".
- **Forced-answer caveat** (§4.1): a forced answer is "accuracy under
  compulsion", not ground truth about the original hedge.

### 14.4 Reporting discipline

- ECE/Brier with bootstrap CIs on the test split.
- Both dose axes reported where relevant: nominal (stated) vs realised
  (calibrated P(correct)) (§8 · 1–2).
- Omniscience-Index reported alongside reliability/resolution (§8.2).
- Provenance per result: seed, config hash, model version, code commit.
- Calibration split used for calibration fitting only, never for training or
  selection.

## 15. Predicted results

> **Read this before the figures.** Everything here is predicted, drawn so
> the analysis is committed in advance and a null is visibly distinguishable
> from a positive. Figures are generated by the stub scripts in `plots/` and
> labelled `predicted`; they are re-run on real data unchanged.

### Figure 1 — calibration curves per signal `predicted under H1`

Calibrated P(correct) vs stated confidence, one curve per signal (verbalized,
behavioral, internal), with a bootstrap CI band. Predicted: the verbalized
curve sits above the diagonal (runs hot) while behavioral and internal hug
it. Null drawn as dashed: all three curves on the diagonal with overlapping
bands. Stub: `plots/fig1_calibration_prediction.py`.

### Figure 2 — quadrant plot `predicted under H2`

Scatter of calibrated verbal vs behavioral/internal, with quadrant counts and
the abstention split (justified hedge vs missed knowledge, §4.1) as a
companion. Predicted: "hopeful" and "suppressed" quadrants cluster by question
type. Null drawn as dashed: uniform scatter, no clustering, examples drawn
randomly across types. Stub: `plots/fig2_quadrant_prediction.py`.

### Figure 3 — base vs Instruct delta `predicted under H3`

Hopeful-confidence rate and missed-knowledge rate side by side for
Qwen2.5-7B (base) vs Qwen2.5-7B-Instruct, with CIs. Predicted: hopeful rate
drops without a matched rise in missed knowledge. Null drawn as dashed:
overlapping bars / equal rates. Stub: `plots/fig3_model_delta_prediction.py`.

### Figure 4 — depth curves `predicted under H4`

AUROC vs. layer percentile, one line per tier, faceted by model size
(§6 · 7). Predicted: retrieval tiers (R1/R2/R3) flat and high from ~0% depth;
reasoning tiers (C1/C2/C3) at chance until late layers, with onset shifting
earlier as scale grows. Null drawn as dashed: all tiers flat at chance,
overlapping bands.

> **Stub not yet written (open item, §0).** The pre-registered form of this
> figure does not exist yet — a `plots/fig4_depth_prediction.py` stub must be
> committed before any run per §17.3 process rules.

## 16. Checkpoints

Go/no-go gates with pre-committed pass rules and named fallbacks. Estimator
and pass rule are fixed before any data is seen.

### Gate 1 — Grading sanity (fires before the full pipeline)

- **Experiment.** E0.
- **Pass rule.** Automated grading agrees with the manual check on ≥95% of
  50–100 hand-verified questions (§3).
- **If it fails.** Escalate grading (string match → NLI → LLM-judge) and
  re-run; if still below 95%, report the agreement rate and treat automated
  labels as provisional, flagging downstream numbers.

### Gate 2 — Format agreement (fires after the §4 subset)

- **Experiment.** E1.
- **Pass rule.** All pairwise Spearman ≥ 0.6 across the three formats (§4,
  H0).
- **If it fails.** Report the three formats separately (the disagreement is
  the finding); use the best-ECE format as canonical and state the
  limitation.

### Gate 3 — Probe validity (fires before §6 output is trusted)

- **Experiment.** E4.
- **Pass rule.** For each (tier × model) cell: the probe across the full
  5-percentile sweep beats the label-shuffle null and the surface/embedding
  baseline on the calibration split, with at least one percentile reaching
  AUROC ≥ 0.65.
- **Pre-check (before trusting any AUROC from this gate):** assert every
  extracted activation tensor is finite; log the fraction of non-finite
  values per layer. T4's FP16-only path is a real risk for silent corruption
  in late layers — this turns "the probe doesn't work" from an ambiguous
  result into a diagnosable one (dirty activations vs. genuinely no signal).
- **If it fails.** Burn the finiteness pre-check first; if activations are
  clean, upgrade to a 1-layer MLP / PCA pre-reduction; if still below bar,
  drop the internal signal for that cell and report "probe fails positive
  control", or reduce to the binary label (§6 · 5).

### Gate 4 — Model comparison (fires before the H3 claim)

- **Experiment.** E5, E6.
- **Pass rule.** Base-model elicitation verified (usable output for formats
  A/B/C, §4 — or a documented limitation); then the H3 delta CI excludes 0
  with the missed-knowledge guard (§13 H3).
- **If it fails.** If elicitation fails on the base model: base-only
  characterisation with the limitation documented. If the delta is within
  noise: report the honest null — both outcomes are reportable (§12).

### Standing risks (re-ranked after each run)

1. **T4 FP16 non-finite activations** in late layers silently corrupt the
   probe → Gate 3 finiteness pre-check; store extracted vectors in float32
   and log the non-finite fraction (§10).
2. **30-cell compute budget** vs. the 30 GPU-hr/week cap at 2000
   questions/cell → re-check before committing all cells; a ragged grid is
   an acceptable outcome (§10).
3. **Base-model verbalized elicitation** fails or degrades on Qwen2.5-7B-base
   → modified prompt or documented limitation (§9 flag).
4. Dataset (any cell) too small for trustworthy calibration → drop magnitude
   comparison, keep Spearman + raw-score quadrants, state as a scope
   limitation (§8).
5. Forced-answer compulsion changes output — the §4.1 caveat is not optional.
6. Grading noise dominates the mismatch quadrants → kills H2.

## 17. Experiments

Two tables, cross-referenced. **17.1 is the pre-registered design table** —
what we committed to. **17.2 is the run log** — what actually ran, filled in
after each run and marked with a verdict.

### 17.1 Design table (pre-registered)

| ID | Experiment | Measure | Tests | Gate |
|---|---|---|---|---|
| E0 | Dataset selection + grading sanity (§3) | manual vs automated agreement on 50–100 hand-verified questions | data quality | Gate 1 |
| E1 | Format agreement (§4) | pairwise Spearman across formats A/B/C on the 50–100-question subset | H0 | Gate 2 |
| E2 | Cell pilot + ladder main run (§3, §4–§8) | 100-question pilot keeps each cell in the 25–80% accuracy band; then all three signals over the surviving cells' train/cal/test split across the full ladder × model grid; per-cell Murphy decomposition + pooled hierarchical regression | H1, H2, H4 | — |
| E3 | Forced-answer companion (§4.1) | forced-answer pass on every Format C Pass → justified-hedge vs missed-knowledge split | H2 | — |
| E4 | Probe validity (§6) | 5-percentile sweep on the calibration split; finiteness pre-check; label-shuffle null; surface/embedding baseline; train AUROC | H2, H4 | Gate 3 |
| E5 | Base-model elicitation check (§9) | Qwen2.5-7B-base usable output for formats A/B/C; document limitation if not | infra | Gate 4 |
| E6 | Post-training comparison run (§9) | repeat E2/E3/E4 on 7B-base; hopeful-confidence and missed-knowledge deltas vs 7B-Instruct | H3 | Gate 4 |

### 17.2 Run log (what actually ran)

Filled in after each run. Run-log IDs may drift from design IDs; when they do,
each row names the design ID it addresses.

| Run-log ID | What it did | Outcome | Headline |
|---|---|---|---|
| *(empty until runs start)* | | | |

### 17.3 Process rules

- **A committed `run.py` before any number.** An ad-hoc run is unreproducible
  and its own results get retracted.
- **Pre-commit expected results — including expected degeneracies** — in the
  run docstring. A degeneracy predicted in advance is recognisable as an
  artefact; one that is not gets written up as a finding.
- **Every activation measurement ships a surface / prompt-only baseline.**
- **Measure the noise floor before interpreting any Δ.** A difference without
  a denominator is a number, not a measurement.
- **Every long Kaggle job checkpoints and resumes by question ID** (§10);
  a job that cannot resume is not a measurement.
