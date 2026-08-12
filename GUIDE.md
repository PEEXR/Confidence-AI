# GUIDE.md — Research Document Structure

A structural blueprint for the confidence project (`PLAN.md` v3), derived
from `calibration-program.html`. The final revision (v3) merged
`CHANGESforPLANv3.md` into v2: six-tier retrieval→reasoning dataset ladder
(§3), Qwen2.5 same-family model ladder + 7B base-vs-Instruct pair (§9),
Kaggle 2× T4 hardware (§10), five-percentile depth probing (§6), and the
Murphy-decomposition / pooled-hierarchical-regression analysis rework (§8).
The
calibration page is a **pre-registration that survived contact with data**:
the plan sections were written before any experiment ran, and a reconciliation
section (`§00`) was added afterward that marks superseded claims *in place*
rather than silently editing them. That two-layer design is the single most
important thing to copy. Everything below is the skeleton plus the rules that
make it work.

---

## 1. The skeleton (16 sections)

| # | Section | Function | Written when |
|---|---------|----------|--------------|
| 00 | **Status — what survived contact with data** | Reconciliation layer: headline numbers, how each pre-commit died, retractions | After experiments |
| 01 | **The problem** | The gap, with a diagram of the circularity/weakness you attack | Before |
| 02 | **Framing** | The obvious objection, stated at full strength, and why it is harmless if the claim is scoped correctly | Before |
| 03 | **Design** | Axes, why the chosen design beats the naive one, the non-circularity rule | Before |
| 04 | **Organisms** | The constructed systems with known ground truth: construction, ground truth, manipulation check | Before |
| 05 | **Hypotheses** | One card per hypothesis: prediction, falsification condition, why it matters | Before |
| 06 | **Instruments** | The measurement battery, each with a predicted loading on each axis | Before |
| 07 | **Experiments** | The design table (pre-registered) **and** the run log (what actually ran) | Both |
| 08 | **Controls** | Measurement nulls, robustness, manipulation checks, reporting discipline | Before |
| 09 | **Ablations** | Causal perturbations, each mapped to the question it answers | Before |
| 10 | **Predicted results** | Pre-registration figures — drawn so a null is visibly distinguishable from a positive | Before |
| 11 | **Later phases** | Beyond the window; explicitly out of scope, with wrong-vs-actual reasons | Before |
| 12 | **Risks, checkpoints, pivots** | Go/no-go gates with pass rules and named fallbacks; standing risks | Before |
| 13 | **Literature** | Sources organised by *role in the program*, each tagged with what it supplies/forecloses | Before |
| 14 | **Glossary** | Every term that carries argumentative weight, defined | Before |
| 15 | **Timeline** | Lanes, sequencing notes, what the final day delivers | Before |

The `#` column doubles as the URL anchor (`#hypotheses`). Keep the numbering
stable even after edits — the calibration doc explicitly keeps v1 numbers so
anything quoting it still resolves.

---

## 2. §00 — the reconciliation layer (why the doc is worth trusting)

Everything that follows is pre-registration. §00 is where the plan meets
reality. Its rules:

- **Headline the current state in 6 lines or fewer**, before any narrative.
- **Lead with numbers as statistics cards** — including the failures
  (`4 / 4 gate candidates failed`), not just the successes.
- **Mark superseded claims in place, never delete them.** Wrong predictions
  are the part with evidential value. Use a visually distinct "retraction"
  block (border colour + label) so the reader sees the claim as written *and*
  what actually happened, side by side.
- **For each failure, record how it died** — one row per pre-committed
  candidate, with the verdict and the mechanism, not just the verdict.
- **Track what survived** explicitly (two columns: *holds up* vs *open /
  not yet tested*).
- **When a measurement turns out to have a noise floor you did not measure,
  say that** — retroactively rescore earlier numbers against it.
- **Self-flag methodological errors** before a reviewer does. The doc
  flags its own seed-fitting mistake ("the configuration was fitted to seed
  0") in the open.

Pattern: every factual claim that later moved carries a visible marker
(`retracted`, `superseded`, `vindicated`, `untested — blocked`) and a link
back to the §00 subsection that explains it. Nothing is silently rewritten.

---

## 3. §01–§02 — problem and framing

**§01 must**:
- State the field's current argument, then the gap in it, in a few bullets.
- Include a **figure of the circularity/weakness** (a diagram beats prose for
  this).
- Close with a **callout: "the move"** — the one-sentence statement of what
  your design does differently.

**§02 must**:
- State **the strongest objection, verbatim, at full strength** — not a
  strawman.
- Admit where the objection is correct, then show it is harmless **given the
  claim you are actually making**. The calibration doc's version: "you are
  not measuring welfare, you are measuring *which axis each instrument
  tracks*; the map does not adjudicate."
- Include **the load-bearing row**: an argument that survives every theory
  on offer. For the calibration doc it is "an instrument whose readout is
  identical across two constructed systems is uninformative regardless of
  which axis you care about." Find your equivalent and make it explicit.
- End with the **language rule** (what you may and may not say in the
  write-up). E.g. calibration: never say "we measured welfare" or "the
  instrument failed"; say "responds to functional structure, to narration, or
  to both."

---

## 4. §03–§04 — design and constructed systems

**§03 (design)** must contain:
- The axes you manipulate, defined with inline tokens (e.g. `function` vs
  `narration`).
- **Why the chosen design beats the naive one** — an explicit "why not the
  simple version" block. Calibration: binary 2×2 gives one `d′` per
  instrument ("detects something"); dose-response gives a curve per axis
  ("measures something").
- A **design figure** showing the actual cell structure.
- The **design consequence of a named hypothesis** — if you add a cell to
  test an interaction, justify it *in terms of the hypothesis*. Calibration:
  the centre point exists because H4 says the axes are *not* additive, and a
  design that assumes additivity cannot test its failure.
- **Nominal vs realised dose.** The calibration doc's rule: plot against
  *realised* exposure (what was actually induced) as primary, *intended*
  exposure as the manipulated variable — and **report both axes** so a reader
  who rejects one can re-plot.
- The **non-circularity rule** — the sentence that keeps the whole design
  honest. Calibration: "every instrument is measured out-of-domain; nothing
  that observes the training environment's policy counts." State your
  equivalent early and repeat it where it bites.

**§04 (constructed systems)** — a table per system with four columns:
`ID | construction | ground truth | manipulation check`. Then:
- For each system, an explicit **confound analysis**: what else differs
  besides the axis of interest, and how you control it. Calibration's A′
  exists because A is RL and B is SFT — so A-vs-B would be confounded with
  training method. A method-matched control is the fix.
- **Axioms stated as axioms.** Where a ground-truth claim cannot be proved
  from inside the design (it rests on a manipulation check that is itself a
  measurement), say so explicitly and say what would falsify it.
- For every axis, **enumerate the things that must be held fixed** while it
  varies (grammatical person, salience, deception — whatever your analogue
  is).
- **A stated "if this system cannot be built" branch** — the failure of the
  construction is itself a result (it fires a hypothesis).

---

## 5. §05 — hypotheses

One card per hypothesis, each with exactly these fields:

- **ID + tier + scope** (`H0 · tier 1 · no training required`).
- **Statement** — one sentence.
- **Predicts** — the specific observation.
- **Falsified if** — the specific observation that kills it. *Every*
  hypothesis must have a falsifier; a hypothesis without one does not belong
  in the document.
- **Why it matters** — what changes in the field if it is true.
- **Estimator and pass rule, fixed pre-hoc.** Calibration's rule of thumb is
  the best line in the doc: *"an estimator chosen after seeing the curves is
  not an estimator."* Pre-commit the statistic (e.g. standardised slope with
  a bootstrap CI) and the pass rule (CI on one loading excludes the other) —
  not "top-left quadrant," which is eyeballing.
- Reserve **H0 as the abort-branch deliverable**: the thing that needs no
  training, no constructed system, and no gate, and is publishable alone.
  Every plan needs one — it is what makes the null honourable instead of a
  scramble.
- A **status callout** at the head of the section, updated after runs: which
  hypotheses are tested, blocked, untouched.

---

## 6. §06 — instruments (your three signals)

The battery table: one row per instrument, columns
`instrument | family | what it reads | loading on axis 1 | loading on axis 2`.

- **Loading numbers are predictions, not measurements** — say so in a note
  under the table ("listed so the design is falsifiable in advance").
- Family labels matter: they let the reader see at a glance which rows are
  expected to behave alike.
- Every instrument needs a **positive control rule**: a condition in which it
  *must* fire, or it is excluded from the map rather than plotted at the
  origin. An instrument that cannot clear its own positive control is
  reporting "no signal at this scale," not "fails the screen" — the doc
  insists these are opposite conclusions.
- Include a **cut rule** for time pressure: cut instruments, never cells or
  seeds — "four instruments across the full cross at three seeds gives real
  curves with variance; eleven instruments at one seed gives a slope through
  three points with no denominator, which is not a measurement."

---

## 7. §07 — experiments

Two tables, explicitly cross-referenced:

1. **The design table** (pre-registered): `ID | experiment | measure |
   tests | window`. This is what you *committed to*.
2. **The run log** (what actually ran): `run-log ID | what it did | outcome |
   headline`. Every row carries a verdict-coloured outcome and a one-line
   headline with the actual numbers.

The collision hazard: **the design IDs and the run-log IDs may drift apart**
(design `E5` ≠ run-log `E5`). The calibration doc solves this with a warning
box and a convention: run-log IDs are written with their directory name
(`E5_class_balance_control`), design IDs with a section reference, and an
*unqualified* "E5" is defined as ambiguous.

Rules the run log enforced (worth stealing as process rules):
- **A committed `run.py` before any number.** The one experiment driven ad
  hoc was unreproducible and its own results had to be retracted.
- **Pre-commit expected results — including expected degeneracies** — in the
  run docstring. This is what turns a spectacular `−1.000000` from a
  confirmation into a recognised artefact.
- **Every activation measurement ships a surface / prompt-only baseline.**
- **Measure the noise floor before interpreting any difference.**

---

## 8. §08–§09 — controls and ablations

**Controls** are grouped into four cards. Steal the grouping wholesale:
- **Measurement nulls** — the baselines that must be beaten: norm-matched
  random direction, bag-of-words lexical baseline (does a cheap surface
  model reproduce the effect?), label-shuffle null for probes, random-choice
  baseline for coherence claims, prompt-only classifier for any
  self-report/privileged-access claim.
- **Robustness & construct** — persona swap, position/first-token bias null,
  test–retest at fixed temperature, evaluation-awareness probe, out-of-domain
  enforcement audit.
- **Manipulation checks** — the checks that verify each constructed system
  has the ground truth you claim. Mark the single most important one; the
  whole design rests on it, so instrument it from day one.
- **Reporting discipline** — TPR at fixed FPR, pre-registered grader rubrics,
  ≥3 seeds per cell reported with a CI, both dose axes reported, disclose the
  pre-built vs in-window split.

Annotate each control that *fired* (caught a real failure) — the doc
explicitly marks the prompt-only baseline as "the only control that fired"
and promotes it from a self-report-only check to a requirement for *every*
activation measurement.

**Ablations** are a table: `ID | ablation | question it answers | priority`.
- Every ablation must be mapped to a *specific* question (the "half of the
  H2 dissociation" style), not "interesting to check."
- Label core / high / medium / post-sprint so the reader knows what can be
  cut.
- Note reclassifications where an earlier design gave a component the wrong
  job ("a confound in the main contrast has to be designed out, not checked
  afterwards").

---

## 9. §10–§11 — predicted results and later phases

**Predicted results** must:
- Open with a callout: **"Read this before the figures"** — everything here
  is predicted, drawn so the analysis is committed in advance and a null is
  visibly distinguishable from a positive.
- Carry a `predicted under H#` marker on every figure. The falsifying
  alternatives are drawn *in* the figure (dashed lines) with their
  implications labelled.
- Include the ablation prediction table too, with a **null row** — the
  outcome where both readouts move together is a real, pre-committed result.

**Later phases** must:
- Be kept out of the current window's commitments, and **explicitly out of
  scope** items listed with a two-column table: *the wrong reason for cutting
  it* vs *the actual reason*. This is where you keep the reasoning on the
  record that the obvious justification is not the right one.

---

## 10. §12–§15 — risks, literature, glossary, timeline

**Risks / checkpoints** — a sequence of go/no-go gates, each a block with:
- The day it fires.
- The pass rule (pre-committed, e.g. "≥4/6 seeds clearing `rate < 0.75 ×
  random`, tuned on held-out seeds only").
- **If it fails:** the named fallback or pivot, in order of preference.
- A **standing risks** list that is honestly re-ranked after each run (the
  calibration doc's number one risk was scooping; after the runs it became
  "no valid gate exists").

**Literature** — organised by *role in the program*, not chronology:
`substrate | instrument sources | constraints | method precedent | normative
frame`. Each entry tagged with what it *supplies* (role) and what it
*forecloses* (kills as a standalone project). Include a **provenance
warning**: "verified earlier in a conversation is not a citation" — flag
which quoted numbers are load-bearing and unverified, and require re-pulling
every identifier before submission.

**Glossary** — every term carrying argumentative weight, including the
constructs you coin. Two-column grid, term + definition, with the program's
own definitions first.

**Timeline** — lanes (compute / engineering / gates) on a day grid, with the
dependency structure explicit (compute should never sit idle), plus the
"what the final day delivers" list — including what is **not** delivered and
said so.

---

## 11. Cross-cutting conventions to hold everywhere

1. **Pre-register the estimator and the pass rule before any data.** An
   estimator chosen after the curves are seen is not an estimator.
2. **Pre-commit expected results, including expected degeneracies.** The
   degeneracy that is predicted in advance is recognisable as an artefact;
   the one that is not gets written up as a finding.
3. **Every instrument gets a condition in which it must fire** (positive
   control) before its readout means anything.
4. **Measure the noise floor before any Δ** — repeat the measurement on an
   unchanged system until it has an SD. A difference without a denominator is
   a number, not a measurement.
5. **Never let the trained system choose the evaluation set.**
6. **Tune on held-out seeds only.** Fitting hyperparameters to the reporting
   seed and then reporting that seed is fitting the result.
7. **Superseded claims are marked in place, never deleted** — that is what
   makes a pre-registration a research record rather than a plan.
8. **Be honest about which numbers are predictions.** Every predicted loading
   / figure / curve is labelled as such.
9. **Keep section numbering stable** once quoted anywhere.
10. **Every reported result carries provenance** (seed, config hash, code
    SHA) so a failure can be reproduced or retracted.

---

## 12. Mapping to this project (PLAN.md v3)

The calibration doc's components map to the confidence project as follows.
Use the section skeleton above with these substitutions (section references are
to v3):

| Calibration doc | This project |
|---|---|
| **Ground truth by construction** | Correct/incorrect labels (PLAN §7); grading sanity check (PLAN §3) is the analogue of a manipulation check |
| **Instruments (battery)** | The three signals: verbalized, behavioral, internal (PLAN §4–§6). "Family" → signal type |
| **Loading map (fn × nar)** | Quadrant analysis: calibrated verbal vs behavioral/internal (PLAN §8 · 4) — the load-bearing output |
| **Hypotheses** | H0: three-format agreement (PLAN §4) — the no-training, publishable-alone deliverable; H1: the three signals have systematically different calibration; H2: mismatch cases are systematic, not noise, and characterisable; H3: post-training (Qwen2.5-7B-Instruct vs 7B-base, PLAN §9) reduces the "hopeful confidence" rate; H4: probe signal onsets later (deeper) for reasoning than retrieval tiers, and earlier as scale grows (PLAN §6, §13) |
| **Design cells** | Six-tier ladder (PLAN §3) × model ladder + 7B-base-vs-Instruct pair (PLAN §9) × sampling temperature (PLAN §5) × percentile sweep (PLAN §6). Format arms A/B/C (PLAN §4) apply within each cell. Cells are committed by pilot (25–80% band) + GPU budget — the grid is intentionally ragged (PLAN §3, §10) |
| **Positive controls** | Format-agreement check (PLAN §4); probe must beat a label-shuffle null and a surface/embedding baseline; semantic-entropy sanity cases (PLAN §5 · 4) |
| **Measurement nulls** | Bag-of-words/lexical sentiment direction for any "valence-like" claim; prompt-only classifier; label-shuffle null for the probe; random-choice baseline for any coherence claim |
| **Manipulation checks** | The bucket→probability mapping must be *empirical* (actual accuracy per bucket, PLAN §4), never hand-assigned; grading must clear the ~5% manual-agreement bar (PLAN §3); the forced-answer caveat — "accuracy under compulsion", not ground truth about the original hedge — is stated in §4.1 |
| **Ablations** | Drop a signal from the comparison; probe trained on held-out questions only (leakage control); percentile sweep (PLAN §6 · 3); post-training comparison 7B-base vs 7B-Instruct (PLAN §9). Note: "full vs LoRA FT" is moot — v3 commits to no fine-tuning anywhere (PLAN §10) |
| **Nominal vs realised dose** | Nominal = stated confidence; realised = calibrated P(correct) against ground truth — report both (PLAN §8 · 1–2) |
| **Decision-level metric** | Omniscience-Index (PLAN §8.2) as a companion to the Murphy decomposition (reliability/resolution, PLAN §8) — the analogue of the calibration doc's "TPR at fixed FPR" reporting-discipline metric |
| **Cut rule** | Cut formats/models, never questions or the calibration/test split |
| **Falsifier discipline** | Each quadrant (PLAN §8 · 4) needs a *named falsifier* (e.g. H2 falsified if mismatch quadrants are dominated by grading noise rather than question type) |
| **§00 reconciliation** | Where the 7B-base-vs-Instruct comparison result (PLAN §9), the abstention split (PLAN §4.1), the depth-curve findings (PLAN §6, H4), and any retracted claim land — written after runs, marked in place |

### Current v3 status
`CHANGES.md` (v1→v2) and `CHANGESforPLANv3.md` (v2→v3) are merged; that diff
is resolved. PLAN keeps its own 12-section organisation (research question →
deliverable) rather than adopting the skeleton above; that is fine — the
skeleton is a checklist, not a requirement. The five gaps the skeleton
originally flagged are now closed:

- **§0 Status** — present, one line ("pre-registration, no runs yet") plus a
  list of the three open items carried into v3.
- **A hypotheses section** — H0–H4 are each a card with a "falsified if"
  clause (§13). H4 (depth-of-signal onsets later for reasoning, earlier with
  scale) is new to v3.
- **Controls** — consolidated in §14 (nulls · robustness · manipulation ·
  reporting), including the percentile-sweep-on-calibration-split-only rule
  and the activation-finiteness check.
- **Predicted results** — Figures 1–4 (§15), each labelled `predicted under
  H#` with the null drawn in. Figure 4 (depth curves, H4) is the one not yet
  pre-registered as a stub — a `plots/fig4_depth_prediction.py` stub must be
  committed before any run.
- **Checkpoints** — Gates 1–4 (§16) with pre-committed pass rules and named
  fallbacks; Gate 3 includes the T4-FP16 finiteness pre-check.

Still open in v3 (tracked as standing risks, §16): the Fig 4 stub above; the
30-cell compute budget vs the 30 GPU-hr/week Kaggle cap; and whether
Qwen2.5-7B-base yields usable verbalized-confidence output (E5, §9).

---

## 13. Final checklist for any document following this guide

- [ ] §00 reconciliation section exists and is honest about failures
- [ ] Every hypothesis has a "falsified if" clause
- [ ] Estimator and pass rule are fixed pre-hoc, stated in the doc
- [ ] Every instrument has a positive-control condition
- [ ] Noise floor / null SD measured before any Δ is claimed
- [ ] Prediction figures are labelled `predicted under H#`, with falsifier
      alternatives drawn in
- [ ] Controls grouped: nulls · robustness · manipulation checks · reporting
- [ ] Ablations each answer one named question
- [ ] A published abort branch (H0) needs no training and stands alone
- [ ] Superseded claims are marked in place, never silently edited
- [ ] Provenance warning on literature; load-bearing quoted numbers flagged
- [ ] Section numbering stable; every ID unambiguous
# GUIDE.md — Research Document Structure

A structural blueprint for the confidence project (`PLAN.md` v2), derived
from `calibration-program.html`. The four items that used to live in
`CHANGES.md` were merged into v2 (abstention granularity → §4.1,
AA-Omniscience → §3, LFM2.5-8B-A1B → §9, Omniscience-Index → §8.2). The
calibration page is a **pre-registration that survived contact with data**:
the plan sections were written before any experiment ran, and a reconciliation
section (`§00`) was added afterward that marks superseded claims *in place*
rather than silently editing them. That two-layer design is the single most
important thing to copy. Everything below is the skeleton plus the rules that
make it work.

---

## 1. The skeleton (16 sections)

| # | Section | Function | Written when |
|---|---------|----------|--------------|
| 00 | **Status — what survived contact with data** | Reconciliation layer: headline numbers, how each pre-commit died, retractions | After experiments |
| 01 | **The problem** | The gap, with a diagram of the circularity/weakness you attack | Before |
| 02 | **Framing** | The obvious objection, stated at full strength, and why it is harmless if the claim is scoped correctly | Before |
| 03 | **Design** | Axes, why the chosen design beats the naive one, the non-circularity rule | Before |
| 04 | **Organisms** | The constructed systems with known ground truth: construction, ground truth, manipulation check | Before |
| 05 | **Hypotheses** | One card per hypothesis: prediction, falsification condition, why it matters | Before |
| 06 | **Instruments** | The measurement battery, each with a predicted loading on each axis | Before |
| 07 | **Experiments** | The design table (pre-registered) **and** the run log (what actually ran) | Both |
| 08 | **Controls** | Measurement nulls, robustness, manipulation checks, reporting discipline | Before |
| 09 | **Ablations** | Causal perturbations, each mapped to the question it answers | Before |
| 10 | **Predicted results** | Pre-registration figures — drawn so a null is visibly distinguishable from a positive | Before |
| 11 | **Later phases** | Beyond the window; explicitly out of scope, with wrong-vs-actual reasons | Before |
| 12 | **Risks, checkpoints, pivots** | Go/no-go gates with pass rules and named fallbacks; standing risks | Before |
| 13 | **Literature** | Sources organised by *role in the program*, each tagged with what it supplies/forecloses | Before |
| 14 | **Glossary** | Every term that carries argumentative weight, defined | Before |
| 15 | **Timeline** | Lanes, sequencing notes, what the final day delivers | Before |

The `#` column doubles as the URL anchor (`#hypotheses`). Keep the numbering
stable even after edits — the calibration doc explicitly keeps v1 numbers so
anything quoting it still resolves.

---

## 2. §00 — the reconciliation layer (why the doc is worth trusting)

Everything that follows is pre-registration. §00 is where the plan meets
reality. Its rules:

- **Headline the current state in 6 lines or fewer**, before any narrative.
- **Lead with numbers as statistics cards** — including the failures
  (`4 / 4 gate candidates failed`), not just the successes.
- **Mark superseded claims in place, never delete them.** Wrong predictions
  are the part with evidential value. Use a visually distinct "retraction"
  block (border colour + label) so the reader sees the claim as written *and*
  what actually happened, side by side.
- **For each failure, record how it died** — one row per pre-committed
  candidate, with the verdict and the mechanism, not just the verdict.
- **Track what survived** explicitly (two columns: *holds up* vs *open /
  not yet tested*).
- **When a measurement turns out to have a noise floor you did not measure,
  say that** — retroactively rescore earlier numbers against it.
- **Self-flag methodological errors** before a reviewer does. The doc
  flags its own seed-fitting mistake ("the configuration was fitted to seed
  0") in the open.

Pattern: every factual claim that later moved carries a visible marker
(`retracted`, `superseded`, `vindicated`, `untested — blocked`) and a link
back to the §00 subsection that explains it. Nothing is silently rewritten.

---

## 3. §01–§02 — problem and framing

**§01 must**:
- State the field's current argument, then the gap in it, in a few bullets.
- Include a **figure of the circularity/weakness** (a diagram beats prose for
  this).
- Close with a **callout: "the move"** — the one-sentence statement of what
  your design does differently.

**§02 must**:
- State **the strongest objection, verbatim, at full strength** — not a
  strawman.
- Admit where the objection is correct, then show it is harmless **given the
  claim you are actually making**. The calibration doc's version: "you are
  not measuring welfare, you are measuring *which axis each instrument
  tracks*; the map does not adjudicate."
- Include **the load-bearing row**: an argument that survives every theory
  on offer. For the calibration doc it is "an instrument whose readout is
  identical across two constructed systems is uninformative regardless of
  which axis you care about." Find your equivalent and make it explicit.
- End with the **language rule** (what you may and may not say in the
  write-up). E.g. calibration: never say "we measured welfare" or "the
  instrument failed"; say "responds to functional structure, to narration, or
  to both."

---

## 4. §03–§04 — design and constructed systems

**§03 (design)** must contain:
- The axes you manipulate, defined with inline tokens (e.g. `function` vs
  `narration`).
- **Why the chosen design beats the naive one** — an explicit "why not the
  simple version" block. Calibration: binary 2×2 gives one `d′` per
  instrument ("detects something"); dose-response gives a curve per axis
  ("measures something").
- A **design figure** showing the actual cell structure.
- The **design consequence of a named hypothesis** — if you add a cell to
  test an interaction, justify it *in terms of the hypothesis*. Calibration:
  the centre point exists because H4 says the axes are *not* additive, and a
  design that assumes additivity cannot test its failure.
- **Nominal vs realised dose.** The calibration doc's rule: plot against
  *realised* exposure (what was actually induced) as primary, *intended*
  exposure as the manipulated variable — and **report both axes** so a reader
  who rejects one can re-plot.
- The **non-circularity rule** — the sentence that keeps the whole design
  honest. Calibration: "every instrument is measured out-of-domain; nothing
  that observes the training environment's policy counts." State your
  equivalent early and repeat it where it bites.

**§04 (constructed systems)** — a table per system with four columns:
`ID | construction | ground truth | manipulation check`. Then:
- For each system, an explicit **confound analysis**: what else differs
  besides the axis of interest, and how you control it. Calibration's A′
  exists because A is RL and B is SFT — so A-vs-B would be confounded with
  training method. A method-matched control is the fix.
- **Axioms stated as axioms.** Where a ground-truth claim cannot be proved
  from inside the design (it rests on a manipulation check that is itself a
  measurement), say so explicitly and say what would falsify it.
- For every axis, **enumerate the things that must be held fixed** while it
  varies (grammatical person, salience, deception — whatever your analogue
  is).
- **A stated "if this system cannot be built" branch** — the failure of the
  construction is itself a result (it fires a hypothesis).

---

## 5. §05 — hypotheses

One card per hypothesis, each with exactly these fields:

- **ID + tier + scope** (`H0 · tier 1 · no training required`).
- **Statement** — one sentence.
- **Predicts** — the specific observation.
- **Falsified if** — the specific observation that kills it. *Every*
  hypothesis must have a falsifier; a hypothesis without one does not belong
  in the document.
- **Why it matters** — what changes in the field if it is true.
- **Estimator and pass rule, fixed pre-hoc.** Calibration's rule of thumb is
  the best line in the doc: *"an estimator chosen after seeing the curves is
  not an estimator."* Pre-commit the statistic (e.g. standardised slope with
  a bootstrap CI) and the pass rule (CI on one loading excludes the other) —
  not "top-left quadrant," which is eyeballing.
- Reserve **H0 as the abort-branch deliverable**: the thing that needs no
  training, no constructed system, and no gate, and is publishable alone.
  Every plan needs one — it is what makes the null honourable instead of a
  scramble.
- A **status callout** at the head of the section, updated after runs: which
  hypotheses are tested, blocked, untouched.

---

## 6. §06 — instruments (your three signals)

The battery table: one row per instrument, columns
`instrument | family | what it reads | loading on axis 1 | loading on axis 2`.

- **Loading numbers are predictions, not measurements** — say so in a note
  under the table ("listed so the design is falsifiable in advance").
- Family labels matter: they let the reader see at a glance which rows are
  expected to behave alike.
- Every instrument needs a **positive control rule**: a condition in which it
  *must* fire, or it is excluded from the map rather than plotted at the
  origin. An instrument that cannot clear its own positive control is
  reporting "no signal at this scale," not "fails the screen" — the doc
  insists these are opposite conclusions.
- Include a **cut rule** for time pressure: cut instruments, never cells or
  seeds — "four instruments across the full cross at three seeds gives real
  curves with variance; eleven instruments at one seed gives a slope through
  three points with no denominator, which is not a measurement."

---

## 7. §07 — experiments

Two tables, explicitly cross-referenced:

1. **The design table** (pre-registered): `ID | experiment | measure |
   tests | window`. This is what you *committed to*.
2. **The run log** (what actually ran): `run-log ID | what it did | outcome |
   headline`. Every row carries a verdict-coloured outcome and a one-line
   headline with the actual numbers.

The collision hazard: **the design IDs and the run-log IDs may drift apart**
(design `E5` ≠ run-log `E5`). The calibration doc solves this with a warning
box and a convention: run-log IDs are written with their directory name
(`E5_class_balance_control`), design IDs with a section reference, and an
*unqualified* "E5" is defined as ambiguous.

Rules the run log enforced (worth stealing as process rules):
- **A committed `run.py` before any number.** The one experiment driven ad
  hoc was unreproducible and its own results had to be retracted.
- **Pre-commit expected results — including expected degeneracies** — in the
  run docstring. This is what turns a spectacular `−1.000000` from a
  confirmation into a recognised artefact.
- **Every activation measurement ships a surface / prompt-only baseline.**
- **Measure the noise floor before interpreting any difference.**

---

## 8. §08–§09 — controls and ablations

**Controls** are grouped into four cards. Steal the grouping wholesale:
- **Measurement nulls** — the baselines that must be beaten: norm-matched
  random direction, bag-of-words lexical baseline (does a cheap surface
  model reproduce the effect?), label-shuffle null for probes, random-choice
  baseline for coherence claims, prompt-only classifier for any
  self-report/privileged-access claim.
- **Robustness & construct** — persona swap, position/first-token bias null,
  test–retest at fixed temperature, evaluation-awareness probe, out-of-domain
  enforcement audit.
- **Manipulation checks** — the checks that verify each constructed system
  has the ground truth you claim. Mark the single most important one; the
  whole design rests on it, so instrument it from day one.
- **Reporting discipline** — TPR at fixed FPR, pre-registered grader rubrics,
  ≥3 seeds per cell reported with a CI, both dose axes reported, disclose the
  pre-built vs in-window split.

Annotate each control that *fired* (caught a real failure) — the doc
explicitly marks the prompt-only baseline as "the only control that fired"
and promotes it from a self-report-only check to a requirement for *every*
activation measurement.

**Ablations** are a table: `ID | ablation | question it answers | priority`.
- Every ablation must be mapped to a *specific* question (the "half of the
  H2 dissociation" style), not "interesting to check."
- Label core / high / medium / post-sprint so the reader knows what can be
  cut.
- Note reclassifications where an earlier design gave a component the wrong
  job ("a confound in the main contrast has to be designed out, not checked
  afterwards").

---

## 9. §10–§11 — predicted results and later phases

**Predicted results** must:
- Open with a callout: **"Read this before the figures"** — everything here
  is predicted, drawn so the analysis is committed in advance and a null is
  visibly distinguishable from a positive.
- Carry a `predicted under H#` marker on every figure. The falsifying
  alternatives are drawn *in* the figure (dashed lines) with their
  implications labelled.
- Include the ablation prediction table too, with a **null row** — the
  outcome where both readouts move together is a real, pre-committed result.

**Later phases** must:
- Be kept out of the current window's commitments, and **explicitly out of
  scope** items listed with a two-column table: *the wrong reason for cutting
  it* vs *the actual reason*. This is where you keep the reasoning on the
  record that the obvious justification is not the right one.

---

## 10. §12–§15 — risks, literature, glossary, timeline

**Risks / checkpoints** — a sequence of go/no-go gates, each a block with:
- The day it fires.
- The pass rule (pre-committed, e.g. "≥4/6 seeds clearing `rate < 0.75 ×
  random`, tuned on held-out seeds only").
- **If it fails:** the named fallback or pivot, in order of preference.
- A **standing risks** list that is honestly re-ranked after each run (the
  calibration doc's number one risk was scooping; after the runs it became
  "no valid gate exists").

**Literature** — organised by *role in the program*, not chronology:
`substrate | instrument sources | constraints | method precedent | normative
frame`. Each entry tagged with what it *supplies* (role) and what it
*forecloses* (kills as a standalone project). Include a **provenance
warning**: "verified earlier in a conversation is not a citation" — flag
which quoted numbers are load-bearing and unverified, and require re-pulling
every identifier before submission.

**Glossary** — every term carrying argumentative weight, including the
constructs you coin. Two-column grid, term + definition, with the program's
own definitions first.

**Timeline** — lanes (compute / engineering / gates) on a day grid, with the
dependency structure explicit (compute should never sit idle), plus the
"what the final day delivers" list — including what is **not** delivered and
said so.

---

## 11. Cross-cutting conventions to hold everywhere

1. **Pre-register the estimator and the pass rule before any data.** An
   estimator chosen after the curves are seen is not an estimator.
2. **Pre-commit expected results, including expected degeneracies.** The
   degeneracy that is predicted in advance is recognisable as an artefact;
   the one that is not gets written up as a finding.
3. **Every instrument gets a condition in which it must fire** (positive
   control) before its readout means anything.
4. **Measure the noise floor before any Δ** — repeat the measurement on an
   unchanged system until it has an SD. A difference without a denominator is
   a number, not a measurement.
5. **Never let the trained system choose the evaluation set.**
6. **Tune on held-out seeds only.** Fitting hyperparameters to the reporting
   seed and then reporting that seed is fitting the result.
7. **Superseded claims are marked in place, never deleted** — that is what
   makes a pre-registration a research record rather than a plan.
8. **Be honest about which numbers are predictions.** Every predicted loading
   / figure / curve is labelled as such.
9. **Keep section numbering stable** once quoted anywhere.
10. **Every reported result carries provenance** (seed, config hash, code
    SHA) so a failure can be reproduced or retracted.

---

## 12. Mapping to this project (PLAN.md v2)

The calibration doc's components map to the confidence project as follows.
Use the section skeleton above with these substitutions (section references are
to v2):

| Calibration doc | This project |
|---|---|
| **Ground truth by construction** | Correct/incorrect labels (PLAN §7); grading sanity check (PLAN §3) is the analogue of a manipulation check |
| **Instruments (battery)** | The three signals: verbalized, behavioral, internal (PLAN §4–§6). "Family" → signal type |
| **Loading map (fn × nar)** | Quadrant analysis: calibrated verbal vs behavioral/internal (PLAN §8 · 4) — the load-bearing output |
| **Hypotheses** | e.g. H1: the three signals have systematically different calibration; H2: mismatch cases are systematic, not noise, and characterisable; H3: abstention-trained model (LFM2.5-8B-A1B, PLAN §9) reduces the "hopeful confidence" rate vs baseline; H0: the three-format agreement check (PLAN §4) is the no-training, publishable-alone deliverable |
| **Design cells** | Format arms (A/B/C, PLAN §4) × questions (PLAN §3) × model (baseline vs LFM, PLAN §9) × sampling temperature (PLAN §5) |
| **Positive controls** | Format-agreement check (PLAN §4); probe must beat a label-shuffle null and a surface/embedding baseline; semantic-entropy sanity cases (PLAN §5 · 4) |
| **Measurement nulls** | Bag-of-words/lexical sentiment direction for any "valence-like" claim; prompt-only classifier; label-shuffle null for the probe; random-choice baseline for any coherence claim |
| **Manipulation checks** | The bucket→probability mapping must be *empirical* (actual accuracy per bucket, PLAN §4), never hand-assigned; grading must clear the ~5% manual-agreement bar (PLAN §3); the forced-answer caveat — "accuracy under compulsion", not ground truth about the original hedge — is stated in §4.1 |
| **Ablations** | Drop a signal from the comparison; probe trained on held-out questions only (leakage control); layer sweep (PLAN §6 · 3); model comparison baseline-vs-LFM (PLAN §9). Note: "full vs LoRA FT" is now moot — v2 commits to no fine-tuning anywhere (PLAN §10) |
| **Nominal vs realised dose** | Nominal = stated confidence; realised = calibrated P(correct) against ground truth — report both (PLAN §8 · 1–2) |
| **Decision-level metric** | Omniscience-Index (PLAN §8.2) as a companion to ECE/Brier — the analogue of the calibration doc's "TPR at fixed FPR" reporting-discipline metric |
| **Cut rule** | Cut formats/models, never questions or the calibration/test split |
| **Falsifier discipline** | Each quadrant (PLAN §8 · 4) needs a *named falsifier* (e.g. H2 falsified if mismatch quadrants are dominated by grading noise rather than question type) |
| **§00 reconciliation** | Where the LFM comparison result (PLAN §9), the abstention split (PLAN §4.1), and any retracted claim land — written after runs, marked in place |

### Suggested next step
`CHANGES.md` is merged into v2, so that diff is resolved. v2 keeps its own
12-section organisation (research question → deliverable) rather than adopting
the skeleton above; that is fine — the skeleton is a checklist, not a
requirement. What v2 still lacks, relative to the skeleton, is five things:

- **§00 Status** — add a short section at the top, initially one line:
  "pre-registration, no runs yet." It becomes the reconciliation layer the
  moment a number exists.
- **A hypotheses section** — PLAN describes *what* will be measured (signals,
  quadrant analysis, model delta) but never states each hypothesis as a card
  with a "falsified if" clause. H0 (three-format agreement), H1 (differential
  calibration), H2 (mismatch cases are systematic and characterisable), H3
  (abstention training reduces hopeful confidence) make the doc falsifiable
  in advance.
- **Controls** — a consolidated list. Today the controls live scattered inside
  §4–§8 (format-agreement check, ~5% grading bar, label-shuffle null, layer
  sweep on the calibration split only, calibration-split discipline). One
  grouped list makes it auditable.
- **Predicted results** — the figures you commit to now: (1) calibration
  curves per signal, (2) the quadrant plot, (3) the baseline-vs-LFM delta,
  each labelled `predicted` and each drawn so the null is distinguishable from
  a positive.
- **Checkpoints** — the grading sanity check (§3), the format-agreement gate
  (§4), and the model-comparison gate (§9) each need a pre-committed pass rule
  and a named fallback. The calibration doc's rule of thumb: the estimator and
  pass rule are fixed before any data is seen.

---

## 13. Final checklist for any document following this guide

- [ ] §00 reconciliation section exists and is honest about failures
- [ ] Every hypothesis has a "falsified if" clause
- [ ] Estimator and pass rule are fixed pre-hoc, stated in the doc
- [ ] Every instrument has a positive-control condition
- [ ] Noise floor / null SD measured before any Δ is claimed
- [ ] Prediction figures are labelled `predicted under H#`, with falsifier
      alternatives drawn in
- [ ] Controls grouped: nulls · robustness · manipulation checks · reporting
- [ ] Ablations each answer one named question
- [ ] A published abort branch (H0) needs no training and stands alone
- [ ] Superseded claims are marked in place, never silently edited
- [ ] Provenance warning on literature; load-bearing quoted numbers flagged
- [ ] Section numbering stable; every ID unambiguous
