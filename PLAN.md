# Genuine vs. Hopeful Confidence in LLMs — Research Plan (v2)

> **Revision note (Aug 2026):** This version merges all four items from
> CHANGES.md into the base plan, and updates Section 9/10 with the
> model-selection decisions from the hardware sizing discussion (full
> precision small model instead of a quantized larger one — rationale below).

## 0. Status

**Status: pre-registration — no runs yet.**

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

Headline metric: for each signal, how well-calibrated is it against ground
truth (ECE, Brier score), and how much do the three disagree with each other
(Spearman correlation + quadrant analysis of mismatches)?

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
small model with characterization of the disagreement cases.

## 3. Dataset

- **PopQA** (preferred — has built-in popularity/difficulty score) or
  **SimpleQA** (adversarially filtered, good error rate for interesting
  calibration curves).
- **AA-Omniscience** (Artificial Analysis) as a third option — 6,000
  questions, domain-tagged, explicitly designed to separate correct /
  incorrect / abstained rather than just correct/incorrect. Likely a better
  fit than PopQA for the calibration angle specifically.
  - Check whether the public HuggingFace subset has enough coverage and
    question variety for the 300–500 question target; if too thin, fall
    back to PopQA/SimpleQA, or mix both (stratify by domain from
    AA-Omniscience, backfill volume from PopQA).
  - Domain tags from AA-Omniscience (Business, Humanities & Social Sciences,
    Law, Health, Science/Engineering/Math, Software Engineering) can be
    reused directly as the stratification variable instead of building one
    from popularity scores.
- 300–500 questions, stratified across the difficulty range (or by domain,
  if using AA-Omniscience).
- Split: **60% train / 20% calibration / 20% test** (or k-fold if the set
  ends up too thin to split three ways reliably).
- **Sanity check before running the full pipeline:** hand-verify correctness
  on 50–100 questions. If automated grading disagrees with manual check on
  more than ~5%, fix grading before trusting downstream numbers.

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

1. **Single greedy forward pass** per question (temperature=0),
   `output_hidden_states=True`. This is deliberately cheap — one pass, not
   ten — since the point is testing whether internals recover the behavioral
   signal without the sampling cost.
2. **Extract hidden state** at the **last prompt token** position (start
   here — cleaner than last-generated-token, no look-ahead bias from the
   model's own answer).
3. **Sweep layers** (e.g. every 3rd–4th layer across depth) — don't assume
   which layer wins, it's model-dependent. Usually mid-to-late layers.
4. **Train logistic regression per layer** on (hidden_state → correct/
   incorrect), with `StandardScaler` first. Use a 1-layer MLP or PCA
   pre-reduction only if logistic regression clearly underfits (train
   AUROC < ~0.65).
5. **Label options:**
   - Binary: ground-truth correctness
   - Continuous: the semantic entropy value from Signal 2 (this is the
     actual Semantic Entropy Probes approach — more interesting since it
     directly tests "does a single forward pass recover the 10-sample
     behavioral signal?")
6. **Select best layer** by AUROC on the **calibration** split (never on
   train).
7. **Calibrate** the winning probe's raw output via isotonic regression
   fit on the calibration split, applied to the test split.

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
3. **Metrics to report:**
   - ECE and Brier score per signal against ground truth (test split)
   - Pairwise Spearman correlation between the three raw signals (doesn't
     require calibration — rank-order only)
   - Pairwise Pearson correlation between the three *calibrated* signals
     (magnitude comparison — requires calibration to be meaningful)
   - **Omniscience-Index** (decision-level companion metric — see 8.2)
4. **Quadrant / mismatch analysis** — pull actual examples:
   - High verbal + low behavioral/internal → **"hopeful" / performed
     confidence** (likely headline finding)
   - Low verbal + high behavioral/internal → suppressed confidence /
     hedging despite real knowledge (possible safety-training artifact)
   - **Missed knowledge** cases from Section 4.1 fold into this same
     low-verbal/high-internal quadrant via the Format C betting frame —
     report alongside the verbal-hedging cases, not as a separate category
   - Characterize what kinds of questions land in each quadrant (long-tail
     entities, dates, ambiguous phrasing, etc.)
5. **Per-model comparison** (baseline vs. comparison model, Section 9): run
   1–4 above for each model separately, then report deltas — specifically,
   does the abstention-trained comparison model show a lower rate of
   high-verbal/low-behavioral-or-internal ("hopeful confidence") cases than
   the baseline? Report ECE/Brier for both models side by side; the delta
   between them is the actual finding, not just two separate result tables.

**If the dataset is too small for calibration to be trustworthy:** drop
magnitude comparison, keep Spearman correlation + raw-score quadrant
bucketing, and state this as an explicit scope limitation in the writeup —
that's a legitimate call, not a flaw.

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
- Report per model (baseline vs. comparison, Section 9) and per domain if
  using AA-Omniscience's domain tags.
- Companion summary statistic, not a replacement for ECE/Brier — keep both.

**Why abstain = 0 and not split into +1/−1:** a benchmark score can only
grade what was actually observed. An abstained question has no produced
answer, so there's no ground-truth comparison to make — scoring it would
require the forced-answer pass from Section 4.1. Where that data exists (via
the Format C forced-answer companion), use the finer-grained justified/missed
split from 4.1 for the qualitative analysis; keep the Index itself at the
standard +1/−1/0 scoring so it stays comparable to published AA-Omniscience
numbers for other models.

## 9. Models

Two models, run through the full pipeline (Sections 4–8) independently, then
compared:

**Baseline — no explicit abstention training:**
- **Qwen3.5-0.8B** (preferred) — full precision (FP16/BF16), no
  quantization. ~1.9GB VRAM for weights + KV cache, leaving ample headroom
  for `output_hidden_states=True` extraction on an 8GB card.
- **Llama 3.2 1B Instruct** (alternative) — full precision, ~3GB VRAM for
  weights + KV cache. Slightly more world knowledge than the 0.8B Qwen
  model at the cost of a bit less headroom.
- Either is deliberately small. The research question is about *signal
  disagreement*, not model capability — see Section 9.1 for the reasoning
  behind choosing small-and-full-precision over larger-and-quantized.

**Comparison — abstention-trained:**
- **LFM2.5-8B-A1B** (Liquid AI) — explicitly RL-trained for abstention
  (avg@k-based reward reinforcing abstention beyond reliable knowledge).
  Running the full three-signal pipeline on it turns the project from pure
  characterization into a real comparison: does abstention-RL training
  reduce the "hopeful confidence" mismatch rate compared to a model that
  never received that training?
- Mixture-of-Experts, ~1B active params — confirm `output_hidden_states`
  extraction works cleanly through the MoE routing before assuming
  Section 6's probe pipeline ports over unchanged. Routing may affect which
  layer/position carries the clearest signal, so a **fresh layer sweep is
  needed** rather than reusing whatever layer won for the baseline model.
- ~1B active params should run comfortably on the RX 6600, likely more
  comfortably than the baseline for the N=10 sampling step in Section 5.

### 9.1 Why small + full precision instead of large + quantized

Signal 3's linear probes train directly on hidden states — quantization
noise corrupts that feature space in ways that are hard to distinguish from
genuine calibration signal. Since accuracy is not the outcome being
measured (disagreement *between* signals is), a smaller full-precision
model gives cleaner, more interpretable activations than a larger quantized
one, without sacrificing anything the research question actually needs.
Smaller models also tend to show clearer overconfidence effects, which
likely makes the quadrant mismatches in Section 8 more visible rather than
weaker.

## 10. Hardware / compute notes

- **No fine-tuning anywhere in this pipeline.** Both models run in
  inference-only mode across all three signals; the only things "trained"
  are the small auxiliary models in Sections 6 and 8 (logistic regression,
  isotonic/Platt calibration) fit on extracted features on CPU — negligible
  VRAM cost, no gradients touch the base model. This avoids the ~4x
  gradient/optimizer-state overhead fine-tuning would otherwise add.
- **VRAM budget (RX 6600, 8GB), full precision, no quantization:**
  - Qwen3.5-0.8B: ~1.9GB weights + KV cache → ~6GB headroom for hidden-state
    extraction.
  - Llama 3.2 1B: ~3GB weights + KV cache → ~5GB headroom.
  - LFM2.5-8B-A1B (MoE, ~1B active): comparable or better headroom than the
    dense baseline for the N=10 sampling step.
- The **N=10 sampling step** for semantic entropy (Section 5) is the
  expensive part regardless of model size — batch it, keep question count
  modest (few hundred) until the pipeline is validated end-to-end, then
  scale up if compute allows.
- `output_hidden_states=True` is the memory-spiking step (storing all
  layers × seq_len × hidden_dim) regardless of model size. Use
  `torch.cuda.empty_cache()` between batches, and drop to batch size 1–2 or
  per-layer extraction (rather than storing all layers at once) if hitting
  OOM — unlikely at this model scale, but worth guarding against on longer
  sequences.

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

- **Statement.** Abstention-RL training reduces "hopeful" confidence without
  trading it for blanket hedging.
- **Predicts.** LFM2.5-8B-A1B shows a lower rate of hopeful-confidence cases
  (high verbal, low behavioral/internal) than the baseline Qwen3.5-0.8B, and
  the reduction is not offset by a rise in "missed knowledge" (suppressed)
  cases (§4.1).
- **Falsified if.** The delta in hopeful-confidence rate between models has a
  CI including 0, or abstention is explained by blanket hedging (the
  missed-knowledge guard triggers).
- **Why it matters.** Turns pure characterisation into a comparison: is
  calibration trainable, and does the training do the right thing rather than
  just hedge more?
- **Estimator + pass rule.** Bootstrap CI on the difference of
  hopeful-confidence rates (baseline − LFM) on matched question sets; pass if
  the lower bound > 0 AND the missed-knowledge rate does not rise by ≥ the
  amount hopeful confidence falls.
- **Experiment.** E5, E6.

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

- **Layer sweep on the calibration split only** (§6): the winning layer is
  selected on calibration, never on train or test.
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
- Omniscience-Index reported alongside ECE/Brier (§8.2).
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

### Figure 3 — baseline vs LFM delta `predicted under H3`

Hopeful-confidence rate and missed-knowledge rate side by side for baseline
vs LFM, with CIs. Predicted: hopeful rate drops without a matched rise in
missed knowledge. Null drawn as dashed: overlapping bars / equal rates. Stub:
`plots/fig3_model_delta_prediction.py`.

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
- **Pass rule.** The winning layer (selected on the calibration split) beats
  the label-shuffle null and a surface/embedding baseline, and the logistic
  train AUROC ≥ 0.65.
- **If it fails.** Upgrade to a 1-layer MLP / PCA pre-reduction; if still
  below bar, drop the internal signal and report "probe fails positive
  control", or reduce to the binary label (§6 · 5).

### Gate 4 — Model comparison (fires before the H3 claim)

- **Experiment.** E5, E6.
- **Pass rule.** `output_hidden_states` extraction verified through LFM MoE
  routing with a fresh layer sweep (§9); then the H3 delta CI excludes 0 with
  the missed-knowledge guard.
- **If it fails.** If MoE extraction fails: baseline-only characterisation.
  If the delta is within noise: report the honest null — both outcomes are
  reportable (§12).

### Standing risks (re-ranked after each run)

1. Dataset too small for trustworthy calibration → drop magnitude comparison,
   keep Spearman + raw-score quadrants, state as a scope limitation (§8).
2. MoE routing breaks the probe pipeline (§9).
3. Forced-answer compulsion changes output — the §4.1 caveat is not optional.
4. Grading noise dominates the mismatch quadrants → kills H2.

## 17. Experiments

Two tables, cross-referenced. **17.1 is the pre-registered design table** —
what we committed to. **17.2 is the run log** — what actually ran, filled in
after each run and marked with a verdict.

### 17.1 Design table (pre-registered)

| ID | Experiment | Measure | Tests | Gate |
|---|---|---|---|---|
| E0 | Dataset selection + grading sanity (§3) | manual vs automated agreement on 50–100 hand-verified questions | data quality | Gate 1 |
| E1 | Format agreement (§4) | pairwise Spearman across formats A/B/C on the 50–100-question subset | H0 | Gate 2 |
| E2 | Baseline main run (§4–§8) | all three signals over the full train/cal/test split; per-signal ECE/Brier, quadrant counts | H1, H2 | — |
| E3 | Forced-answer companion (§4.1) | forced-answer pass on every Format C Pass → justified-hedge vs missed-knowledge split | H2 | — |
| E4 | Probe validity (§6) | layer sweep on the calibration split; label-shuffle null; surface/embedding baseline; logistic train AUROC | H2 | Gate 3 |
| E5 | LFM MoE extraction check (§9) | `output_hidden_states` extraction verified through MoE routing; fresh layer sweep | infra | Gate 4 |
| E6 | LFM main run (§9) | repeat E2/E3/E4 on the comparison model; hopeful-confidence and missed-knowledge deltas vs baseline | H3 | Gate 4 |

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