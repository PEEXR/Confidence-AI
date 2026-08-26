# Audit Tasks Checklist

Cross-referenced against [`AUDIT.md`](AUDIT.md) on 2026-08-26. Every item below
traces to a numbered audit finding or a prioritized improvement.

**Legend**

- `[x]` — implemented in `confidence_pipeline.py` **and** covered by a test in
  this repo's audit test suite.
- `[~]` — code is complete and tested, but the **artefacts still carry
  pre-repair numbers**. Closing the item needs a production GPU run.
- `[ ]` — not started.
- `[✓]` — **answered by data.** The repaired analysis has run over the existing
  generations and the item has an empirical result, not just working code.

**Result note (2026-08-26).** `audit_docs/reanalyse.py` re-ran the CPU analysis
stages over the existing `graded.parquet` with the repaired code. Several items
below are now closed by evidence rather than by code:

- **Gate 2 is falsified** — the three elicitation formats do not correlate
  (A–C ρ = −0.047). They cannot be collapsed to a canonical axis.
- **Format B's ECE is 0.0000 exactly** — the circularity is confirmed in the
  shipped artefact, not only in the source.
- **H3 has 0 testable rows.** Not a null result; the comparison was never
  available.
- **H2's 111 cases come from 2 of 6 cells**, and 4 of those 6 never cross the
  threshold at all.
- **Verbalized confidence is degenerate in 24 of 30 cells**, with nothing usable
  below 3B.

Full tables in [`FINDINGS.md`](FINDINGS.md). **H2 and H3 are withdrawn outright**,
not reframed as descriptive: the axis beneath them is largely absent.

**Status note (2026-08-26).** The previous revision of this file marked twelve
items `[x]` and annotated most others as implemented "in repair-v2". None of
that code was present in `confidence_pipeline.py`, the `.ipynb`, or either
branch — a grep for the 24 symbols those notes name returned zero hits, and all
four blocking defects were still live in the file. Every item was therefore
re-opened and implemented from the pre-repair state. Checkbox states below
reflect the code as it now stands.

---

## H2: separate cell effects from question-feature effects

- [x] Reframe the current descriptive result as: confidence disagreement is heterogeneous across model × tier cells. *(`QUADRANT_TAXONOMY` / `QUADRANT_LABELS` module constants: Performative Certainty / Excessive Hedging / Grounded Certainty / Honest Doubt. Legacy storage keys retained everywhere. H2 output carries `counts_labelled`, `taxonomy`, and an explicit `reframing` field.)*
- [x] Test whether question features (for example, length and numeric content) predict hopeful confidence within cells, rather than only in the pooled dataset. *(`associations_per_cell` runs the χ² inside each cell with n ≥ 40, where tier and model are held constant.)*
- [x] Fit a mixed-effects analysis that accounts for model, tier, and model × tier cell baselines; report both cell-level and pooled estimates. *(`hierarchical_regression` fits random intercepts for question, model, tier AND model × tier, degrading through a documented ladder if the VB fit fails, and falls back to a logit clustered on the CELL rather than on the question. `per_cell` carries the unpooled fits for the shrinkage comparison.)*
- [x] Define and analyze a continuous confidence discrepancy metric $\Delta = \text{Verbal} - \text{Behavioral}$ alongside binary quadrant thresholding. *(`delta_verbal_behavioral` and `delta_verbal_internal` assembled in `assemble_signals`; distribution reported overall and per quadrant in the H2 output.)*
- [x] Replicate any feature association across multiple cells before making a general claim about question characteristics. *(`replicates_across_cells`: significance required in ≥ 2 cells with n ≥ 40. A feature that only survives pooling is reported as a tier proxy, not a question characteristic.)*
- [x] Report how concentrated the result is. *(New. `concentration` gives per-quadrant counts by cell and the top-3-cell share; `threshold_degenerate_cells` names every cell whose `verbal_cal` never crosses the 0.5 cut and which therefore contributes exactly one quadrant by construction — AUDIT finding 6.)*

## H3: repair and re-evaluate the post-training comparison

- [x] Fix NaN coercion bug in `confidence_pipeline.py` (`test_h3_base_vs_instruct`). *(The indicator is built with `np.where` over a validity mask, so a missing verbal or behavioral value stays NaN instead of becoming a hard `0.0`.)*
- [x] Exclude rows with missing verbal or behavioral/internal confidence values before creating the binary indicator; then recompute the matched rates and bootstrap CI. *(Matching now runs over questions both variants scored **validly**, not merely questions both were asked. `n_excluded_missing_signal` and `excluded_by_model` are reported. Below 20 valid matched rows per variant H3 returns `available: False` rather than reporting a rate.)*
- [✓] Re-run the missed-knowledge guard on the repaired matched comparison. *(Moot on this data: H3 returns `available: false` because **0 matched rows** carry both a verbal and a behavioural value — 802 excluded, 600 of them the base model. The guard never runs because the comparison does not exist.)*
- [x] Do not equate the indicator with "confident and wrong"; report its correctness rate and the rate of confident incorrect answers separately. *(`per_model` carries `flagged_rows_correct_rate`, `confident_incorrect_rate` and `overall_correct_rate` for each variant.)*
- [x] If the result replicates, reframe it as a descriptive trade-off. *(Verdict text and the Figure 3 caption rewritten; `fig3_model_delta` draws only when `available`.)*
- [x] Retire the "hopeful" label in H3 outputs/reports. *(Outputs use `unwarranted_rate_base` / `unwarranted_rate_instruct`; `hopeful_rate_*` retained as deprecated aliases so existing artefacts keep loading.)*

## Internal Probing & Activation Tap: fix prompt-boundary capture

- [x] Fix activation tap hook: capture hidden states during a dedicated prompt prefill forward pass *before* calling `generate()`. *(`prefill_capture` runs the prompt through `_transformer_base()` — the decoder stack with no logits head — then `tap.freeze()` before decode. `last_prompt_index` derives the index from the attention mask, so it is correct under left AND right padding. `_transformer_base` raises rather than tapping an unrecognised architecture. **Tested** against a hand-computed reference at all 5 depths under both padding sides; the frozen buffer survives simulated decoding; p0 is provably constant across rows sharing a final token.)*
- [x] Verify the negative control baseline: confirm that layer 0 (`p0` embedding) probe scores AUROC ≈ 0.50 on the constant prompt-ending token. *(`p0_neg_control_auroc` / `_dev` / `_pass` recorded per cell; Gate 3 **blocks** when the deviation exceeds `P0_NEG_CONTROL_TOL` (0.10). Tested on the audit's exact 0.717 case, on a clean 0.52 case, and on the missing-p0 case — which does not pass, since an absent control cannot certify the tap. Verification against real data happens on the next run.)*
- [~] Re-extract internal activations across all 5 depth percentiles on the last prompt token. *(Needs a GPU run.)*
- [~] Re-train and re-evaluate linear correctness probes on the clean pre-generation activations. *(Needs a GPU run.)*
- [x] Activate `PROBE_C_GRID` parameter sweep. *(`select_probe_C` picks C by 3-fold CV **inside the training split**, breaking ties toward stronger regularisation; shuffle-nulls refit at the selected C; the chosen C flows into `internal_scores`, the sweep table and the Gate 3 report. Selection deliberately avoids the calibration split: calibration AUROC also drives `meets_gate`, the winning-percentile choice and Gate 3, so maximising it over the grid and then gating on the maximum is selection on the test statistic — measured on pure noise it moved the false-pass rate at the 0.65 gate from 1.5% to 4.0%. `PROBE_C_SELECT=False` restores the old fixed C=1.0.)*
- [~] Re-run the depth-of-decodability analysis (H4) and recompute `internal_cal` signals across all evaluated cells. *(Needs a GPU run.)*
- [x] Promote the depth-of-decodability result out of personal notes into README/headline results. *(README now leads with it, with an explicit banner that the figures come from the pre-repair tap and must be regenerated.)*

## Verbal Axis: replace label-derived Format B with raw stated confidence

- [x] Make Format A's parsed value actually be the model's stated number. *(New, surfaced by verification. `parse_confidence_numeric` divided by 100 unconditionally, so a model answering `0.9` was recorded as **0.009** — a hundredfold error that lands inside [0,1], is a distinct value so the degeneracy pre-flight cannot see it, and reads as near-total uncertainty from a model expressing near-total confidence. Survivable while B was canonical; load-bearing now that A is. The scale is inferred instead: fractions (`7/10`, `7 out of 10`), explicit percentages, decimals at or below 1 taken as probabilities, otherwise the 0-100 scale the prompt asks for.)*
- [x] Replace label-derived empirical accuracy in Format B with genuine model-stated confidence. *(Both routes offered by the audit are implemented: Format A carries the model's own stated number, clipped to [0,1]; and `BUCKET_FIXED_VALUES` gives Format B a label-free reading emitted as variant `Bfix`. `LABEL_FREE_FORMATS = ("A", "Bfix", "C")` is the only set canonical selection draws from, in selection AND in the degenerate fallback, with an assertion behind it. **Tested**: B is not chosen even when constructed to have the lowest Brier of all formats.)*
- [x] Report label-derived Format B only as an auxiliary supervised accuracy baseline. *(B is still scored in `fmt_stats` and `verbal_long`, tagged `"role": "supervised_reference"` in `calibration_meta.json`. H0 / Gate 2 still pairs A/B/C, so Gate 2 semantics are unchanged.)*
- [x] Remove circular double-calibration. *(The calibrator is fit to the canonical label-free signal; B's accuracy-mapped values never reach it.)*
- [✓] Recompute `verbal_raw` across all evaluated cells, fit calibrators cleanly, and evaluate true test ECE / Brier. *(Done on CPU over the existing generations. Canonical axis moves **B → Bfix**; B's ECE is 0.0000 exactly; A 0.5296, Bfix 0.4738, C 0.4248. Removing the circularity costs Brier 0.2197 → 0.4680 — B looked calibrated because it was scored against its own construction.)*

## Semantic Entropy: sequence log-probs, math equivalence, valid-sample normalization

- [x] Compute sequence log-probabilities during sample generation and apply length-normalization / Rao-Blackwellization. *(`sequence_logprobs` scores the sampled sequences in a separate teacher-forced pass, chunked over rows and time → mean per-token log-probability under the **raw model distribution**, stored per sample. Two cheaper routes were measured and rejected: `output_scores=True` retains a `[batch × n_return, vocab]` float tensor per generated token, which at Qwen2.5's ~152k vocabulary is tens of gigabytes for an N=10 sampling batch — larger than the KV cache the batch sizer budgets for, and on some cells larger than the card; and a `LogitsProcessor` is cheap but sees logits with `repetition_penalty` applied and temperature/top-p not, matching neither the model likelihood nor the sampler's distribution. `cluster_mass` weights each cluster by summed length-normalised likelihood; because that is a geometric mean, a smaller cluster can carry more mass, so `modal_share` (counts) and `modal_mass` (probability) are separate columns. A cell falls back to count weighting **wholesale** unless every record has log-probs, so one calibrator is never fit over two scales. **Tested** against a naive per-token reference on real `generate()` output — greedy, N=10 sampled, hot sampling with early stops, and `max_new_tokens=1` — plus invariance across five chunk settings and NaN (never 0.0) for all-stop rows.)*
- [x] Extend semantic clustering to math answer forms. *(`math_equal` reuses `grade_latex` — the same oracle as the grader — bidirectionally, with a numeric fallback so clustering quality does not depend on whether the optional `math_verify` package installed. Union-find merges latex/numeric BEFORE the NLI pass, whose gate now includes them. **Tested**: `0.5 ≡ 1/2 ≡ \frac{1}{2} ≡ 2/4` merge while `0.25` stays separate, with and without `math_verify`; text clustering is unaffected; the merge is transitive.)*
- [x] Fix the valid-sample normalization edge case. *(Both remedies: $H_{\max} = \log N_{\text{requested}}$, and `confidence_behavioral = NaN` when the parsed-sample count is below `ENTROPY_MIN_VALID` (8). `low_valid` recorded per row. **Tested**: n=1 yields NaN, not 1.0.)*
- [✓] Recompute behavioral confidence scores and re-calibrate across all cells. *(Done on CPU, **count-weighted** — log-prob weighting still needs the raw generations. 403 of 13,000 questions (3.1%) fall below `ENTROPY_MIN_VALID` and are now NaN; 16 of them had a single parsed sample and previously scored confidence 1.0. Behavioural entropy out-resolves the verbal axis **4.3×** with full 13/13 cell coverage.)*

## Cross-Signal Commensurability: symmetric 3-signal alignment

- [x] Set `PROBE_LABEL = "entropy"` so Signal 3 tests whether prompt-boundary activations predict behavioral dispersion. *(Default flipped; `correct` retained as a documented supervised reference mode.)*
- [x] Align prompt variants across modalities. *(SAMPLE and EXTRACT already share an `instruction()` branch and a `FEWSHOT_POOL` entry; `assert_prompt_alignment` now verifies byte-identical rendering per tier at runtime and raises on drift, so a future edit cannot silently decouple them.)*
- [x] Reframe the 2×2 disagreement quadrants as an epistemic alignment taxonomy. *(See H2 above; Figure 2's legend and caption are generated through `quadrant_label()`.)*
- [x] Evaluate ground-truth correctness strictly as an external downstream benchmark. *(All three raw signals are label-free: verbal via the label-free format set, probe via the entropy label. Labels enter only through calibrator fitting on the calibration split and test-split scoring.)*
- [x] Score all three signals against **one** ground truth. *(New — AUDIT finding 4. `correct` used to come from the canonical verbal pass while the probe trained on EXTRACT correctness, and accuracy differs by up to 12 points across variants. `GROUND_TRUTH_VARIANT` (default `EXTRACT`, the plain answering pass the activations were tapped from) now defines correctness for every signal; `accuracy_by_variant` and `accuracy_spread_across_variants` are recorded in `calibration_meta.json`.)*

## Grid Completion & Base-Rate Control: modeling over cell deletion

- [x] Remove the 25–80% cell deletion filter. *(`COMMIT_CELLS_OUTSIDE_BAND` now defaults to True: every cell commits, and the pilot accuracy is recorded as the `in_band` / `pilot_accuracy` covariates instead. Setting it False restores the deleting behaviour.)*
- [~] Recover and analyze the popularity gradient (R1 PopQA High vs R2 PopQA Low) and adversarial retrieval (R3 SimpleQA). *(`popularity_contrast` → `popularity_gradient.json`: per-model accuracy and per-signal means across R1/R2/R3, the R1−R2 accuracy drop with a bootstrap CI, and whether calibrated verbal confidence tracks it. Warns when a tier is absent. R2/R3 cells now execute — the data itself needs a GPU run.)*
- [x] Fit a GLMM over questions nested in tiers and models. *(Random intercepts: question, model, tier, model × tier. Fixed effects: the three calibrated signals, `is_reasoning`, `log(params)`, `is_long`, `has_number`, `has_multi_entity`, `in_band`, plus the pre-registered interactions.)*
- [x] Implement within-cell / difficulty-matched comparisons. *(`difficulty_matched_view` → `difficulty_matched.json`. Difficulty is the **cross-model** correctness rate of a question within its tier, so bins are a property of the item rather than of the model being scored; per-bin AUROC / ECE / Brier / Murphy resolution per signal. Tiers with no within-tier difficulty variation — near-0% or near-100% for every model, which is precisely the regime the band used to delete — are named in `degenerate_tiers` with an explicit `limitation` note, because matching is genuinely unavailable there and the base-rate adjustment falls back to the GLMM's cell random intercept.)*
- [ ] **Reach the planned per-cell N, or drop the size axis and say so.** *(New — AUDIT improvement 9. PLAN §3 targets N=2000; `N_PER_CELL` is 1000 and no cell has reached even that. R2, R3 and the 0.5B row commit zero cells in the shipped results. README now states that no size-axis claim should be made until those cells finish. **Requires GPU time, not a code change** — this is the largest open item.)*

## Data Provenance & Gate Integrity

- [x] Verified Gate 1 human grading check sheet. *(Only `results_v2.5/tables/gate1_manual_check_sheet.csv` is filled: 200/200 rows, 195/200 (97.5%) agreement, 5 disagreement notes, all symbolic-grader LaTeX formatting discrepancies. The sheets in `results_v2_flawed/` and `results_prerepair/` remain blank, so the audit's "0/200 in all three trees" stands for those two.)*
- [x] Prevent pipeline runs from overwriting the human-annotated check sheet; automatically ingest it into final reports. *(`export_manual_check_sheet` refuses to overwrite a sheet with any filled `manual_correct` and writes `…NEW_TEMPLATE.csv` beside it instead. `ingest_manual_sheet` parses the usual spreadsheet spellings, excludes blank rows from the denominator, and reports agreement, disagreement notes and per-grader breakdown. `combined_gate1` treats the human sheet as authoritative over the judge, and returns **None** — not False, not True — when neither has data.)*
- [x] Actively enforce Gate 3 before fitting linear probes. *(Two layers. `stage_probe` blocks a cell on stored OR runtime non-finites before any fit and records it in `gate3_blocked_cells.json`; the Gate 3 verdict additionally requires the p0 negative control and clean activations. And the verdict is now honoured on the **data path**: `internal_scores` emits nothing for a failing cell, so a cell whose p0 scored 0.717 can no longer be written into `gate3.json` as failed while simultaneously feeding 200 `internal_cal` values to H1, H3, H4, the GLMM and the quadrants. Skipped cells are listed in `internal_gate3_skipped.json`.)*
- [x] Restore the H2 gate pass conjunct. *(`test_h2_quadrants` takes `gate1_pass` and `gate3`; the pass rule is the conjunction of association, replication, Gate 1 and Gate 3, and the verdict names the failing conjuncts. The Gate 3 conjunct is evaluated **per cell** over the cells actually contributing rows — a grid-wide `any()` would let one clean cell out of thirteen certify probe validity for every other cell, including cells whose p0 control had failed. The judge now runs BEFORE the stats block so the Gate 1 value exists when H2 is evaluated — it did not, previously, which is why the conjunct was dropped.)*
- [x] Bind every Molab / Kaggle run to the real git commit hash. *(`code_sha` resolves `$CODE_SHA` → walk-up `.git` search from the source file and cwd → `UNBOUND-nogit` with a `RuntimeWarning`. Marks the tree `-dirty` when it is. `run_log_rows.md` flags an unbound run in its header.)*
- [x] Update `PLAN.md` §0 / §17.2 run logs with complete provenance metadata. *(§0 rewritten: three runs recorded with dates and status, all headline numbers withdrawn, the four blocking findings listed. §17.2 carries a row per run, marked as reconstructed because their `code_sha` is `nogit`. §17.3 gains a rule that a run without a commit binding is not a measurement. `run_log_rows.md` is now generated with code_sha, config hash, seed, platform, dtype, library versions, cell coverage and GPU-hours.)*

## Opened by the repaired analysis

- [ ] **Run the elicitation ablation before writing anything up.** The surviving cells use 3–4 distinct confidence values on a 0–100 scale. That may indict the *prompt* rather than the model, and if it does, the headline in `FINDINGS.md` §1 is wrong. Test one alternative at 1.5B — a coarser scale (0–10), or reading confidence off token logits instead of parsed text. Small GPU job, far cheaper than the grid.
- [ ] **Rebuild the question bank from HuggingFace** so H2's within-cell feature tests can run. `build_question_bank()` is seeded, so the same items are recoverable — network only, no GPU. This is the last piece of the verbal-axis diagnosis still missing.
- [ ] **Re-run the grid for Signal 3.** The probe is the one axis that might work and has never been measured with a correct tap. Both outcomes are publishable: a working probe pairs a positive result with the negative one; a failing probe completes the negative paper.
- [ ] **Report the formats separately.** Gate 2 is falsified, which triggers the PLAN §16 fallback — A, B and C cannot be collapsed to a canonical axis in the write-up, whatever the selection code picks for internal use.

## Positioning (AUDIT "Novelty" / "Reframing")

- [x] **Cite the prior work that scooped the stated headline.** *(New. README gains a Related work section: DECK ([arXiv:2606.02289](https://arxiv.org/abs/2606.02289), Jun 2026) as the direct structural twin of the quadrant taxonomy, and Mahaut et al. (ACL 2024) for the verbalized / sequence-probability / P(True) / probe comparison on shared items.)*
- [x] **Reposition as a measurement-validity / negative-results paper with a released pipeline.** *(New. README retitled "Three Measures of Confidence Disagree — Format Sensitivity, Sampling Entropy, and Hidden-State Decodability in Small LLMs", leads with the depth-of-decodability result, states the base-rate control as the actual contribution, and carries an Audit status table naming what is withdrawn.)*

---

## Notes from adversarial verification

Two independent verification passes ran against these fixes; both were asked to
refute rather than confirm. Twenty-one defects were found in the first-pass
repairs and are themselves fixed. The ones worth recording, because they were
not obvious:

- **The first log-probability fix was wrong twice.** `output_scores=True` would
  have retained tens of gigabytes of scores and OOM'd the SAMPLE stage on its
  first batch. Replacing it with a `LogitsProcessor` was cheap but measured
  against a real `generate()` it matched neither the model likelihood nor the
  sampler's distribution — transformers applies custom processors *before* the
  sampling warpers, so it read logits with `repetition_penalty` applied and
  temperature/top-p not. The shipped version re-scores the sampled sequences in
  a separate teacher-forced pass, which is exact and version-independent.
- **That pass needed the attention mask.** Without it, every row attended to its
  own left padding and numbered positions from 0 instead of from the first real
  token. Zero-padding rows were exact and padded rows were wrong in proportion
  to their pad count — the error was ~1e-2, small enough to read as noise, and
  it did not cancel within a question.
- **A gate nothing honours is a comment.** Blocking dirty activations before
  fitting was only half of Gate 3 enforcement: the p0 control and the AUROC gate
  were report-only, so a cell could be recorded as failed and still feed 200
  `internal_cal` values downstream.
- **The first fix for mixed log-prob weighting was too strict** — it required
  every sample's log-prob to be finite, including samples the clustering
  discards, which at N=1000 would have silently disabled the weighting for
  every cell.
- **`"50/50"` parsed as 1.0.** The fix for the confidence parser introduced a
  ratio branch, which read the standard English idiom for maximum uncertainty
  as maximum confidence — on the canonical verbal axis, manufacturing exactly
  the unwarranted stated confidence the project measures. The branch now
  requires a denominator of at most 20, so `50/50` falls through to the 0-100
  reading and gives 0.5.
- **Gate 3 enforcement stopped one step short, twice.** First it blocked
  fitting but not the data path; then it blocked `internal_cal` but not H4,
  whose depth curves are the repo's best finding. It also failed *open* when
  no verdict was supplied.
- **Selecting the probe's C on the calibration split inflated the gate it
  feeds.** Calibration AUROC drives `meets_gate`, the layer choice and Gate 3,
  so maximising it over the C grid is selection on the test statistic — measured
  on pure noise, the false-pass rate at the 0.65 gate went 1.5% → 4.0%.
  Selection moved to cross-validation inside the training split.
- **`"21 out of 21"` scored 0.21.** Bounding the ratio denominator fixed
  `50/50`, but anything above the bound fell through to the 0-100 branch,
  which reads the NUMERATOR — wrong by exactly den/100, and plausible rather
  than absent. An out-of-range ratio now returns None, and `50/50` is handled
  as the idiom it is rather than as a ratio (read literally it is 1.0,
  i.e. certainty, the opposite of what the phrase means).
- **Capturing the sign broke stated ranges.** `"90-95%"` matched `-95` and
  became unparsed. Every numeric branch now carries a lookbehind.
- **Gate 3's H4 filter failed open** while `internal_scores` failed closed —
  opposite defaults, so one dropped keyword argument reverted the H4 repair
  silently, and an empty exclusion list then read as "everything passed".
- **The GLMM went singular in the configuration the repair created.**
  `layer_pct` is an internal-probe artefact; when the probe is gone it fills to
  a constant and its interaction becomes collinear. Constant and duplicate
  terms are now dropped and recorded, and the per-cell fits use the same
  surviving signals as the pooled one instead of a hardcoded three.
- **The tests passed while the code was wrong, twice.** A guard grepping for
  `correction=False` was satisfied by a comment; a behavioural Gate 3 check
  passed because its fixture never reached the code path. Every guard is now
  mutation-tested — the defect is reintroduced and the suite must fail. Fifteen
  mutations are covered; see `tests/README.md`. A later pass mutation-tested
  the suite itself and found four more survivors — including one where an
  end-to-end block had been appended below `sys.exit()` and had never executed.
  Thirty-one mutations are covered now, and `assemble_signals` is exercised
  end-to-end rather than re-implemented inline.

## What still requires GPU time

Every `[~]` item above, plus the per-cell N item. The code is complete and
tested; the artefacts in `results*/` are pre-repair and none of their numbers
should be quoted. Set `CODE_SHA` in the environment before the run, or it will
not be traceable to a commit.

## Cross-reference: AUDIT.md coverage

| AUDIT finding | Covered by |
|---|---|
| 1 — activation tap reads the wrong token | Internal Probing §1–2 |
| 2 — verbal axis is label-derived | Verbal Axis, all items |
| 3 — semantic entropy is a shortcut | Semantic Entropy, all items |
| 4 — three signals not commensurable | Cross-Signal, all items (incl. the new single-ground-truth item) |
| 5 — grid completion 13/30 | Grid Completion §1–2, and the new per-cell N item |
| 6 — quadrant result carried by three cells | H2, all items (incl. the new concentration item) |
| 7 — claims that must be withdrawn | H3 §1–2, PLAN §0, README Audit status |
| 8 — documentation vs data | Data Provenance, all items; `PROBE_C_GRID` under Internal Probing |
| 9 — the 25–80% band | Grid Completion §1, §4 |
| Improvement 5 — promote the depth result | Internal Probing, last item |
| Improvement 9 — N per cell / size axis | Grid Completion, last item |
| Improvement 10 — mixed-effects model | H2 §3, Grid Completion §3 |
| Novelty / Reframing | Positioning |
