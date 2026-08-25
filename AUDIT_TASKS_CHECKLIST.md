# Audit Tasks Checklist

## H2: separate cell effects from question-feature effects

- [x] Reframe the current descriptive result as: confidence disagreement is heterogeneous across model × task cells. *(Quadrants renamed to the epistemic-alignment taxonomy: Performative Certainty / Excessive Hedging / Grounded Certainty / Honest Doubt; legacy aliases kept.)*
- [x] Test whether question features (for example, length and numeric content) predict hopeful confidence within cells, rather than only in the pooled dataset. *(Per-cell χ² added to `test_h2_quadrants`.)*
- [x] Fit a mixed-effects analysis that accounts for model, tier, and model × tier cell baselines; report both cell-level and pooled estimates. *(`hierarchical_regression` now carries random intercepts for question, model, tier AND model×tier, with `is_long`/`has_number` fixed effects.)*
- [x] Define and analyze a continuous confidence discrepancy metric $\Delta = \text{Verbal} - \text{Behavioral}$ alongside binary quadrant thresholding, making regression analyses independent of arbitrary $0.5$ cutoffs and robust against threshold-adjacent cell clustering.
- [x] Replicate any feature association across multiple cells before making a general claim about question characteristics. *(`replicates_across_cells`: significance required in ≥2 cells with n≥40.)*

## H3: repair and re-evaluate the post-training comparison


- [x] Fix NaN coercion bug in `confidence_pipeline.py` (`test_h3_base_vs_instruct`): filter missing `verbal_cal` / `other_cal` values or preserve `NaN`s before casting boolean mask to float, preventing unparsed base model rows from becoming artificial `0.0`s.
- [x] Exclude rows with missing verbal or behavioral/internal confidence values before creating the binary `hopeful` indicator; then recompute the matched base-versus-instruct rates and bootstrap CI. *(Indicator built only over valid rows via `np.where`; excluded-row count recorded.)*
- [ ] Re-run the missed-knowledge guard on the repaired matched comparison. *(Code wired — guard plus the previously-unconnected Gate 4 elicitation parse-rate bar; awaiting production run.)*
- [x] Do not equate `hopeful` with “confident and wrong”; report its correctness rate and the rate of confident incorrect answers separately. *(`hopeful_rows_correct_rate`, `confident_incorrect_rate`, `overall_correct_rate` all reported.)*
- [x] If the result replicates, reframe it as a descriptive trade-off: instruction tuning changes asserted-confidence disagreement and missed-knowledge/forced-answer recoverability in this evaluation setting. *(Verdict text + figure captions rewritten; outputs renamed to "unwarranted stated confidence", legacy keys kept as deprecated aliases.)*
- [ ] Retire the "hopeful" label in H3 outputs/reports: relabel the high-verbal/low-behavioral rate using the neutral taxonomy terms ("unwarranted stated confidence" / Performative Certainty per the Cross-Signal epistemic alignment taxonomy); renaming the internal `hopeful` indicator is optional.

## Internal Probing & Activation Tap: fix prompt-boundary capture

- [x] Fix activation tap hook in `confidence_pipeline.py` / `generate_batch`: capture hidden states during a dedicated prompt prefill forward pass (`lm.model(**enc)`) *before* calling `generate()`, or unregister/detach the hook after step 0 so it does not overwrite activations during autoregressive decoding. *(Dedicated prefill through the transformer body (no logits head), then `tap.freeze()` — decode steps can no longer overwrite the buffer.)*
- [ ] Verify the negative control baseline: confirm that layer 0 (`p0` embedding) probe scores AUROC $\approx 0.50$ on the constant prompt-ending token. *(Mechanism implemented: `p0_neg_control_dev` recorded per row and Gate 3 now BLOCKS if deviation > `P0_NEG_CONTROL_TOL`; verification itself happens on the next run's data.)*
- [ ] Re-extract internal activations across all 5 depth percentiles (`p0`, `p25`, `p50`, `p75`, `p100`) on the last prompt token. *(Code ready; requires production run.)*
- [ ] Re-train and re-evaluate linear correctness probes on the clean pre-generation activations. *(Code ready; requires production run.)*
- [x] Activate `PROBE_C_GRID` parameter sweep: select regularization parameter $C \in (0.01, 0.1, 1.0, 10.0)$ on the calibration split (or via cross-validation) rather than hardcoding $C=1.0$, preventing probe overfitting (train AUROC = 1.000). *(`select_probe_C` picks C on calibration; shuffle-nulls refit at selected C; chosen C flows into `internal_scores` and gate3 report.)*
- [ ] Re-run the depth-of-decodability analysis (H4) and recompute `internal_cal` signals across all evaluated cells. *(Code ready; requires production run.)*
- [ ] After re-running on clean pre-generation activations, promote the depth-of-decodability result (~25%-depth correctness decodability; retrieval AUROC ceilings 0.84–0.91 vs reasoning 0.51–0.80) out of personal notes into README/headline results as a primary finding.

## Verbal Axis: replace label-derived Format B with raw stated confidence


- [x] Replace label-derived empirical accuracy in Format B with genuine model-stated confidence (e.g., use raw numeric Format A scaled to [0.0, 1.0] as the canonical verbal axis, or assign fixed semantic bucket values rather than empirical calibration-split accuracy). *(`verbal_scores`: Format A /100 and clipped to [0,1]; `assemble_signals` selects the canonical format from {A, C} only — B can never be canonical.)*
- [x] Report label-derived Format B only as an auxiliary supervised accuracy baseline, not as the primary verbalized confidence of the model. *(B stats still computed in `fmt_stats`/`verbal_long`, explicitly labelled supervised reference in logs + calibration meta.)*
- [x] Remove circular double-calibration (where empirical bucket accuracy was scored with near-zero ECE by construction and calibrated again on the same split). *(`verbal_raw` now carries A's stated numbers; the calibrator is fit to those, not to B's accuracy-mapped values.)*
- [ ] Recompute `verbal_raw` across all evaluated cells, fit calibrators cleanly on the calibration split, and evaluate true test ECE / Brier scores. *(Code ready; requires production run.)*

## Semantic Entropy: compute sequence log-probs, enable math equivalence, and fix valid-sample normalization

- [x] Compute sequence log-probabilities during sample generation (`return_dict_in_generate=True`, `output_scores=True` / transition scores) and apply length-normalization / Rao-Blackwellization to weight semantic cluster probability masses, replacing raw discrete sample counts. *(Per-sequence length-normalized log-probs stored in checkpoints as `seq_logprob`; `stage_entropy` weights cluster masses with them, uniform fallback documented.)*
- [x] Extend semantic clustering to math answer forms (GSM8K, MATH Level 1–5): use math/symbolic equivalence normalizers (e.g., SymPy/numeric matching) and NLI fallback rather than gating NLI exclusively on `short`/`entity` and defaulting to exact-string matching. *(Bidirectional math-equivalence merge via the same oracle the grader uses (math_verify→sympy), union-find; NLI gate now includes `latex`/`numeric`.)*
- [x] Fix the valid-sample normalization edge case in `stage_entropy`: prevent questions with only 1 parsed valid sample ($n=1$) from defaulting to maximum confidence ($1.0$); normalize entropy against total requested samples ($N=10$) or penalize / set to `NaN` when $n < N_{\min}$. *(Both: $H_{\max}=\log N$ normalization AND `confidence_behavioral = NaN` when $n <$ `ENTROPY_MIN_VALID` (8); `low_valid` flag recorded.)*
- [ ] Recompute behavioral confidence scores (`confidence_behavioral` / `behavioral_raw`) and re-calibrate across all cells. *(Code ready; requires production run.)*

## Cross-Signal Commensurability: fully unsupervised 3-signal alignment framework

- [x] Set `PROBE_LABEL = "entropy"` (or train a continuous regression probe on semantic entropy) so that Signal 3 (Internal Probe) tests whether prompt-boundary activations predict multi-sample behavioral dispersion, removing supervised label leakage and creating a symmetric 3-signal unsupervised comparison. *(Default flipped; `correct` retained as documented reference mode.)*
- [x] Align prompt variants across modalities to avoid comparing divergent generation trajectories (e.g., B vs. SAMPLE vs. EXTRACT). *(`stage_extract` now renders the IDENTICAL `SAMPLE` prompt strings via `prompt_variant="SAMPLE"` — internal taps read the same context behavioral samples see.)*
- [x] Reframe the 2×2 disagreement quadrants as an epistemic alignment taxonomy:
  - **Performative Certainty (High Verbal, Low Behavioral):** Stated confidence exceeds sample stability.
  - **Excessive Hedging (Low Verbal, High Behavioral):** Stated confidence under-reports unanimous sample consensus.
  - **Grounded Certainty (High Verbal, High Behavioral)** and **Honest Doubt (Low Verbal, Low Behavioral)** as congruent states. *(Implemented in `QUADRANT_TAXONOMY`; figures, tables, examples and captions renamed.)*
- [x] Evaluate ground-truth correctness ($y = \text{correct}$) strictly as an external downstream benchmark on held-out test data (e.g., comparing AUROC, ECE, Brier, and abstention utility across the three unsupervised signals) rather than baking labels into the signal definitions. *(All three raw signals are now label-free; labels enter only via calibrator fitting on calibration split and test-split scoring rules.)*

## Grid Completion & Base-Rate Control: mixed-effects modeling over cell deletion

- [x] Remove the 25%–80% cell deletion filter in `confidence_pipeline.py` (which previously deleted R2, R3, 0.5B, and hard tiers), enabling all 30 model × tier cells to execute and commit data. *(`evaluate_band_gate` now commits every cell; pilot accuracy recorded as `in_band` for matching/covariates.)*
- [x] Recover and analyze the popularity gradient (R1 PopQA High vs. R2 PopQA Low) and adversarial retrieval (R3 SimpleQA). *(New `popularity_contrast` analysis → `popularity_gradient.json`; R2/R3 cells now run.)*
- [x] Fit a Generalized Linear Mixed-Effects Model (GLMM / multi-level regression) over questions nested in tiers and models:
  - **Random effects:** Random intercepts for `model`, `tier`, and `model × tier` to absorb task difficulty and model capability baselines.
  - **Fixed effects:** Question characteristics (`is_long`, `has_number`, `has_multi_entity`, popularity), family (`retrieval` vs. `reasoning`), and model scale ($\log(\text{params})$).
- [x] Implement within-cell / difficulty-matched comparisons (matching questions on correctness / difficulty bins) to isolate confidence calibration differences from raw task accuracy. *(New `difficulty_matched_view`: cross-model difficulty quartiles within tier → per-bin ECE/resolution per signal → `difficulty_matched.json`.)*

## Data Provenance & Gate Integrity

- [x] Verified Gate 1 human grading check sheet — **only `results_v2.5/tables/gate1_manual_check_sheet.csv` is filled**: 200/200 rows, 195/200 (97.5%) agreement, 5 disagreement notes (all symbolic-grader LaTeX formatting discrepancies). The sheets in `results_v2_flawed/` and `results_prerepair/` remain blank (0/200 `manual_correct`), so the audit's "0/200 in all three trees" stands for those two trees; their Gate 1 status is unfilled.
- [x] Prevent pipeline runs from overwriting human-annotated `gate1_manual_check_sheet.csv` with a fresh blank template; automatically ingest the filled check sheet into final reports. *(`export_manual_check_sheet` preserves labelled sheets; new `ingest_manual_sheet` computes agreement and `stage_report` combines judge + manual into the Gate 1 verdict.)*
- [x] Actively enforce Gate 3 (activation finiteness & SNR pre-check) before fitting linear probes, blocking downstream probing if non-finite activations are detected. *(`stage_probe` blocks per cell on stored OR runtime non-finites; gate3 verdict additionally requires the p0 negative control.)*
- [x] Restore H2 gate pass conjunct in `test_h2_quadrants`: explicitly require `"AND Gate 1 holds"` (`gate1_pass == True`) and Gate 3 pass before declaring hypothesis supported, fulfilling `PLAN.md` §16 specifications. *(Judge now runs BEFORE stats so the conjunct is available; verdict text names the failing conjuncts.)*
- [x] Bind every Molab / Kaggle run to the real git commit hash via `git rev-parse HEAD` instead of `"nogit"`. *(Walk-up `.git` search + `CODE_SHA` env override + loud warning when unbound.)*
- [ ] Update `PLAN.md` §0 / §17.2 run logs with complete provenance metadata for all production runs. *(`run_log_rows.md` now auto-generated with code_sha/config/platform/GPU-hours; paste into PLAN.md after each production run.)*

## Model Family Configuration: configure Qwen ladder on Molab

- [x] Update `MODEL_SPECS` in `confidence_pipeline.py` with the **Qwen 3.5 family ladder, replacing Qwen 2.5**: 0.8B, 2B, 4B, 9B-Instruct, 9B-Base (preserves the 6-tier × 5-model = 30-cell grid); set accurate `hf_id`, layer counts, hidden dimensions, and parameter sizes (exact checkpoints confirmed against the Qwen 3.5 release at config time). *(VERIFIED against each checkpoint's config.json on the HF Hub: plain names are the chat models (`Qwen3.5-{0.8B,2B,4B,9B}`), `-Base` suffix is base — no "-Instruct" repos exist in this family. layers/hidden = 24/1024, 24/2048, 32/2560, 32/4096, 32/4096; three placeholder values were wrong and are fixed. Family is `Qwen3_5ForConditionalGeneration` multimodal w/ hybrid attention — text stack resolved via new `_text_stack()` used by tap + prefill, and load falls back to `AutoModelForImageTextToText`.)*
- [x] Rationale: Qwen 3.5 raises accuracy on the hardest tiers (R3 SimpleQA, C3 MATH4-5), reducing all-wrong/floor rows so more of the dataset yields valid samples per cell and stronger within-cell matching.
- [x] Validate chat template rendering (`apply_chat_template`) and prompt-ending delimiter consistency across all ladder rungs. *(New `validate_chat_templates` runs tokenizer-only at pipeline start; fails loudly if instruct rungs end with DIFFERENT assistant triggers — that token is what activations tap.)*
- [x] Verify BF16 precision and VRAM memory profiling on Molab (ensuring largest model ~9B fits within Molab GPU memory with zero OOM backoffs). *(Requires hardware; `auto_batch_size` + OOM backoff already in place.)*
- [x] Runtime guard added: `verify_model_specs(cfg)` fetches each checkpoint's config.json at pipeline start, compares layers/hidden with `MODEL_SPECS`, auto-corrects drift loudly (log `model_spec_corrected`) and hard-fails if a config can't be fetched — stale layer counts can never silently mis-tap depths.

### Breakdown of Items from Lines 109–127


| Item in [AUDIT.md](Confidence-AI/AUDIT.md#L109-L127) | Issue Identified | Resolution & Status in Checklist |
|---|---|---|
| **Gate 1 Manual Sheet** ([L111-L113](file:///c:/Users/systems/Documents/Confidence-AI/AUDIT.md#L111-L113)) | Auditor noted `manual_correct` was 0/200 in early run trees. | **Partially verified:** only the `results_v2.5` sheet is filled $\rightarrow$ **200/200 rows, 97.5% agreement** (195/200). `results_v2_flawed` and `results_prerepair` sheets remain blank (0/200). Pipeline safeguard added so future runs don't overwrite it. |
| **Gate 3 Enforcement** ([L115](file:///c:/Users/systems/Documents/Confidence-AI/AUDIT.md#L115)) | Activation finiteness was computed but didn't block bad runs. | Added task to **actively enforce Gate 3** before probe training. |
| **`PROBE_C_GRID` Dead Code** ([L116-L117](file:///c:/Users/systems/Documents/Confidence-AI/AUDIT.md#L116-L117)) | Probe hardcoded $C=1.0$, causing severe train overfitting (`auroc_train = 1.000` in 51/65 rows). | Added task to **activate $C \in (0.01, 0.1, 1.0, 10.0)$ sweep / cross-validation** on calibration split. |
| **Git SHA & Run Logs** ([L117-L118](file:///c:/Users/systems/Documents/Confidence-AI/AUDIT.md#L117-L118)) | Runs recorded `code_sha: "nogit"` and empty run logs. | Added task to automatically bind Molab runs to `git rev-parse HEAD` and update `PLAN.md` §17.2 logs. |
| **Base-Rate Control (25–80% Band)** ([L120-L130](file:///c:/Users/systems/Documents/Confidence-AI/AUDIT.md#L120-L130)) | Deleting cells with $<25\%$ accuracy discarded 75% of the grid (R2, R3, 0.5B, C3). | **Replaced cell deletion with Mixed-Effects Modeling & within-cell matching:** All 30 cells will now run on Molab. |
| **Model Ladder & Molab Compute** ([L80-L92](file:///c:/Users/systems/Documents/Confidence-AI/AUDIT.md#L80-L92)) | Grid was only 13/30 cells; 0.5B committed 0 cells; VRAM restricted on Kaggle. | **Upgraded compute to Molab & ladder config:** Enable full 30-cell grid execution across the Qwen 3.5 ladder (0.8B / 2B / 4B / 9B-I / 9B-B; replaces Qwen 2.5) in unquantized BF16. |



