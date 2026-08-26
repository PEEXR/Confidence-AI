# CONFIG reference

Every knob in the pipeline lives in the `Config` dataclass in **cell 3** of
`confidence_pipeline.ipynb`. Nothing below cell 3 needs editing for a normal
run.

Config is **immutable by convention** — override with `replace()` rather than
mutating, so `CFG.hash()` stays an honest fingerprint of what actually ran:

```python
CFG = replace(_BASE_CFG, N_PER_CELL=2000, ONLY_MODELS=("qwen2.5-7b-base",))
```

`CFG.hash()` is a 12-char SHA-256 over the whole config, stamped onto every
artefact. Same hash ⇒ same knobs.

---

## The five knobs that actually matter

If you touch nothing else, touch these.

| Knob | Default | Why it matters |
|---|---|---|
| `SMOKE` (cell 3, outside the dataclass) | `True` | `True` = 5 questions/cell, ~2 min, validates everything end-to-end. **Set `False` for a real run.** |
| `N_PER_CELL` | `1000` | The single biggest cost driver. 1000 ≈ 7.1 GPU-hr for all 5 models on molab. |
| `ONLY_MODELS` | `()` | How you split a long run across 12-hour sessions. |
| `STAGES` | all 14 | Drop stages to re-run just part of the pipeline. |
| `MODEL_EXEC` | `"sequential"` | Leave it. See § Execution policy for why. |

---

## Identity

| Field | Default | Notes |
|---|---|---|
| `RUN_NAME` | `"run01"` | Output directory name. Reuse it to resume; change it to fork a clean run. |
| `SEED` | `20260813` | Seeds sampling, splits, bootstrap, probes, label shuffles. |
| `NOTES` | — | Free text. **Inert** — not read anywhere. |

## Platform

| Field | Default | Notes |
|---|---|---|
| `PLATFORM` | `"auto"` | `auto \| molab \| kaggle \| colab \| local`. Auto-detects from filesystem markers + GPU size. |
| `OUTPUT_ROOT` | `""` | `""` = per-platform default (`/kaggle/working/confidence`, `./confidence_out`, …). |
| `HF_CACHE` | `""` | `""` = per-platform default, deliberately kept **off** the output volume — 40.7 GB of weights would blow Kaggle's 20 GB `/kaggle/working` cap. |
| `HF_TOKEN` | `""` | Or set the env var. Unauthenticated HF works but is rate-limited. |
| `HF_OFFLINE` | `False` | Sets `HF_HUB_OFFLINE=1`. |

## Stage switches

`STAGES` is both the **skip list and the execution order**. Drop a name to
skip it entirely.

```
data → pilot → verbal → forced → sample → extract →
grade → entropy → probe → calibrate → stats → figures → tables → report
```

| Stage | Does | PLAN |
|---|---|---|
| `data` | build the six-tier bank + 60/20/20 splits | §3 |
| `pilot` | N_PILOT/cell → 25–80% band gate → cell commitment | §3 |
| `verbal` | Signal 1, formats A/B/C | §4 |
| `forced` | forced-answer companion on every Format C Pass | §4.1 |
| `sample` | Signal 2, N=10 at T≥0.7 | §5 |
| `extract` | Signal 3, 5-percentile hooks at last prompt token; writes `variant="EXTRACT"` | §6 |
| `grade` | deterministic grading of everything generated | §7 |
| `entropy` | semantic-entropy clustering → behavioral confidence | §5 |
| `probe` | logistic probe per (cell × percentile) + Gate 3 | §6 |
| `calibrate` | per-signal isotonic/Platt on the calibration split | §8 |
| `stats` | H0–H4, Murphy, Spearman, HLR, Omniscience-Index, quadrants | §8, §13 |
| `figures` / `tables` / `report` | artefacts | §15, §17 |

**Useful subsets:**

```python
# H0-only abort branch — publishable alone, no probe, no sampling (PLAN §13)
STAGES=("data","verbal","grade","stats","figures","tables","report")

# Re-run analysis on existing generations (no GPU generation at all)
STAGES=("grade","entropy","probe","calibrate","stats","figures","tables","report")
```

Later stages read checkpoints from disk when their producing stage is skipped,
so the second form works on a previous session's raw output.

## Grid subset

| Field | Default | Notes |
|---|---|---|
| `ONLY_MODELS` / `ONLY_TIERS` | `()` | Whitelist. **Wins over `SKIP_*`** when non-empty. |
| `SKIP_MODELS` / `SKIP_TIERS` | `()` | Blacklist. |

Model keys: `qwen2.5-0.5b-instruct`, `-1.5b-`, `-3b-`, `-7b-instruct`,
`qwen2.5-7b-base`. Tier keys: `R1 R2 R3 C1 C2 C3`.

## Question budgets

| Field | Default | Notes |
|---|---|---|
| `N_PILOT` | `100` | Pilot size per cell for the band gate. |
| `N_PER_CELL` | `1000` | Committed-cell size. PLAN §3 targets 2000; 1000 fits one molab session. |
| `N_AGREEMENT` | `100` | H0 / Gate 2 subset. Drawn from **train only**. |
| `N_MANUAL_CHECK` | `50` | Rows per answer-form in the Gate 1 hand-check sheet. |
| `SPLIT_FRACTIONS` | `(0.6, 0.2, 0.2)` | train / calibration / test. |

Capped by dataset availability — GSM8K test has only 1319 rows, and a short
tier logs a `tier_short` event rather than silently under-delivering.

> **Small-N floors.** The probe needs ≥20 train and ≥10 calibration rows per
> cell; calibration needs ≥10; H1 needs ≥20 test rows per signal. Below
> `N_PER_CELL ≈ 200` those stages legitimately skip.

## Generation

| Field | Default | Notes |
|---|---|---|
| `DTYPE` | `"auto"` | bf16 on Ampere+, fp16 on Turing, fp32 on CPU. Do not quantize — PLAN §9.1 needs clean activations for the probe. |
| `ATTN_IMPL` | `"sdpa"` | Safe on both T4 (sm_75) and Blackwell. **Do not use `flash_attention_2` on T4** — Turing is unsupported. |
| `BATCH_SIZE` | `0` | `0` = auto-size from free VRAM. Halves automatically on OOM. |
| `BATCH_SIZE_CAP` | `256` | Ceiling for the autosizer. Raise on a 96 GB card. |
| `GREEDY_TEMPERATURE` | `0.0` | Formats A/B/C and the extraction pass. |
| `SAMPLE_TEMPERATURE` | `0.8` | Signal 2. **Asserted ≥ 0.7** — T=0 gives entropy 0 always (PLAN §5). |
| `SAMPLE_TOP_P` | `0.95` | |
| `N_SAMPLES` | `10` | PLAN §5 N=10. The dominant cost term. Also the **fixed** entropy denominator: $H_{\max} = \log N_{\text{SAMPLES}}$. |
| `ENTROPY_MIN_VALID` | `8` | Below this many parsed samples, `confidence_behavioral` is `NaN`. Previously entropy was normalised by the number of samples that happened to parse, so a question with one parsed answer scored a perfect `1.0` (AUDIT finding 3, 7.9% of rows). |
| `ENTROPY_LP_WEIGHTED` | `True` | Weight semantic-cluster mass by length-normalised sequence log-probability instead of raw sample counts. |

Log-probs come from a **separate teacher-forced scoring pass**
(`sequence_logprobs`), chunked over rows and time so peak memory stays bounded.
Two cheaper routes were measured and rejected:

- `output_scores=True` makes `generate()` retain one `[batch × n_return,
  vocab]` float tensor *per generated token*. At Qwen2.5's ~152k vocabulary a
  512-token N=10 sampling batch is tens of gigabytes — more than
  `auto_batch_size` budgets for, and on some cells more than the card holds.
  (CELL 10 refuses `output_hidden_states=True` for the same reason.)
- A `LogitsProcessor` is cheap but does not see a well-defined quantity:
  transformers applies custom processors *before* the sampling warpers, so it
  reads logits with `repetition_penalty` applied but not temperature or top-p.
  Measured against both references, it matches neither, and the ordering is an
  implementation detail that can change between versions.

The scoring pass therefore reports the **raw model likelihood**, unwarped by
`SAMPLE_TEMPERATURE` / `SAMPLE_TOP_P` — the quantity the semantic-entropy
literature uses.

**It is not free.** The pass sees `prompt_len + max_new` token positions per
row against `max_new` for the whole decode loop — 2.7× (C3) to 6× (R1) the
positions — so it roughly **doubles the arithmetic of the SAMPLE stage**. In
wall clock that is ~1.35× at batch size 1, where decode is memory-bandwidth
bound and this pass is not, rising to ~2× at the batch sizes `auto_batch_size`
actually picks (7–8 for the 7B models), because at those sizes decode is
compute-bound too and the discount disappears. Peak memory is ~0.55 GB per
chunk at `row_chunk=4`, independent of prompt length. The measured
`compute_ledger` captures the cost; the a-priori budget estimate does not.

Set `ENTROPY_LP_WEIGHTED=False` to skip it entirely and weight by sample counts
— that restores the pre-audit behaviour and the pre-audit cost.

A cell falls back to count weighting **wholesale** if any of its sample records
predate log-prob recording — logged as `entropy_mixed_weighting`. Calibration
and the entropy median are both per-cell, so a cell has to be on one scale;
delete the sample checkpoint to regenerate it. `lp_weighted` is recorded per
row.

Because mass is length-normalised, a cluster with fewer samples can carry more
mass. `modal_share` (sample fraction) and `modal_mass` (probability fraction)
are therefore reported as separate columns.
| `N_FEWSHOT_BASE` | `4` | Exemplars for `7b-base`, which has no chat template. |
| `STOP_ON_DOUBLE_NEWLINE` | `False` | **Inert.** |

## Execution policy

| Field | Default | Notes |
|---|---|---|
| `MODEL_EXEC` | `"sequential"` | `sequential \| resident \| concurrent`. `resident` currently plans identically to `sequential`. |
| `MAX_CONCURRENT_MODELS` | `2` | Only used under `concurrent`. |
| `CONCURRENT_MAX_PARAMS_B` | `4.0` | Models above this **always run alone**. |
| `MODEL_REPLICAS` | `1` | Replicas of the same weights, each taking a disjoint tier slice. |
| `PURGE_WEIGHTS` | `"never"` | `never \| after_model \| after_run`. See below. |
| `EMPTY_CACHE_EVERY_BATCHES` | `4` | `torch.cuda.empty_cache()` cadence during generation. |

### `PURGE_WEIGHTS` — when disk is reclaimed

GPU memory is **always** cleared after every model regardless of this setting.
This knob only controls the on-disk HF snapshot.

| Value | Behaviour | Use when |
|---|---|---|
| `never` | keep every snapshot | molab — disk is effectively unbounded |
| `after_model` | delete a model's snapshot once it finishes its **final** pass | Kaggle's 20 GB cap, where peak disk matters |
| `after_run` | delete the whole hub cache once, at the very end | you want the cleanup but not the re-downloads |

> **Why `after_model` is not the obvious default.** Each model is loaded
> **twice**: once for the pilot pass, then again for verbal/forced/sample/
> extract. The two passes cannot be merged — the band gate needs *every*
> model's pilot graded before any cell can be committed. Purging after pass 1
> would force a full re-download for pass 2, so the purge is deliberately
> suppressed on the pilot pass and fires only on the final one. It is also
> suppressed entirely when `MODEL_REPLICAS > 1`, since the last shard to
> finish would otherwise delete a snapshot its siblings are still reading.

> **Why concurrency mostly doesn't help.** Batched decode is
> memory-bandwidth bound: every step streams the full weight matrix once. Two
> co-resident 7Bs stream 2× the bytes for the same tokens, so they *split*
> throughput rather than adding it, while halving the VRAM left for KV cache.
> Small models are the opposite — a 0.5B never saturates a 96 GB card, so
> co-running several reclaims genuinely idle SMs. That is exactly what
> `CONCURRENT_MAX_PARAMS_B` encodes.
>
> The same argument applies to `MODEL_REPLICAS`. **To speed up one big model,
> raise `BATCH_SIZE`** — amortising one weight read over more sequences beats
> duplicating the weights.

Regardless of mode, GPU memory is cleared and checkpoints flushed after
**every** model.

## Band gate

| Field | Default | Notes |
|---|---|---|
| `ACCURACY_BAND` | `(0.25, 0.80)` | PLAN §3. **Reported, not enforced.** Recorded per cell as the `in_band` covariate. |
| `COMMIT_CELLS_OUTSIDE_BAND` | `True` | Every cell commits. Set `False` to restore the pre-audit deleting behaviour. |

Conditioning cell inclusion on accuracy is a legitimate base-rate control —
per the audit it is this project's best claim to novelty, because every
published cross-signal comparison confounds method signal with item
difficulty. **Deleting cells is the wrong way to implement it.** It removed all
of R2, all of R3, all of 0.5B and 5/6 of C3 — three quarters of the grid — and
the selection ran on a noisy n=100 pilot whose estimate for one cell moved
0.74 → 0.58 between runs (AUDIT finding 9).

The control is kept and the deletion dropped. `in_band` and `pilot_accuracy`
travel with every signal row, and the base rate is now held fixed two ways:
`difficulty_matched_view` bins questions by their **cross-model** correctness
rate within a tier and compares signals inside a bin, and
`hierarchical_regression` carries a random intercept for the model × tier cell.

## Probe (Signal 3)

| Field | Default | Notes |
|---|---|---|
| `PERCENTILES` | `(0, 25, 50, 75, 100)` | Depth taps. `0` = embedding output, `100` = final block. |
| `PROBE_LABEL` | `"entropy"` | `entropy` = median-split semantic entropy (the Semantic Entropy Probes formulation, PLAN §6·5); `correct` = ground truth, retained as a **supervised reference mode**. |

`PROBE_LABEL` defaulted to `"correct"` before the audit. That made the probe a
second supervised accuracy predictor, so the "three-signal comparison" raced
one unsupervised signal (entropy) against two supervised ones — and all three
were then calibrated to P(correct) anyway (AUDIT finding 4). PLAN §6·5
recommends the entropy label; it is now the default, which makes the
comparison symmetric.

With `PROBE_LABEL="correct"`, label source priority is `EXTRACT → FORCED →
SAMPLE`. Only `EXTRACT` is the greedy pass whose activations were actually
tapped, so anything else is label noise relative to its paired feature vector
and logs a `probe_label_fallback` event. Check the log before trusting a probe
AUROC.
| `PROBE_MAX_ITER` | `2000` | Logistic regression iterations. |
| `PROBE_STORE_DTYPE` | `"float32"` | PLAN §16 standing risk 1 — never store activations in half precision. |
| `AUROC_GATE` | `0.65` | Gate 3 threshold and the H4 onset definition. |
| `LABEL_SHUFFLE_REPEATS` | `20` | Null distribution size. Winner must beat its p95. |
| `SURFACE_BASELINE` | `True` | TF-IDF prompt-only control (PLAN §14.1). **Leave on** — §17.3 requires every activation measurement to ship one. |
| `PROBE_C_GRID` | `(0.01, 0.1, 1.0, 10.0)` | Regularisation strengths searched by `select_probe_C`. |
| `PROBE_C_SELECT` | `True` | Pick `C` by 3-fold CV **inside the training split**, ties broken toward the **smaller** C. `False` restores the fixed `C=1.0`. |
| `GATE3_ENFORCE` | `True` | Block probe fitting on cells with non-finite activations, rather than fitting and noting it afterwards. |
| `P0_NEG_CONTROL_TOL` | `0.10` | Gate 3 fails a cell when \|AUROC(p0) − 0.5\| exceeds this. |

`PROBE_C_GRID` was declared and never read: every probe was fit at `C=1.0`,
which on a ~3584-dimensional activation with a few hundred training rows is
barely regularised — hence `auroc_train = 1.000` in 51 of 65 rows (AUDIT
finding 8). The shuffle nulls refit at the **selected** C, since a null fitted
at a different regularisation strength is not a null for that probe.

Selection runs inside TRAIN, not on calibration, even though calibration is
where the winning percentile is chosen. Calibration AUROC is not only a
selection statistic here — the same number drives `meets_gate`, the layer
choice and Gate 3 — so maximising it over four values of C and then gating on
the maximum is selection on the test statistic. Measured on pure noise
(200 trials, n=160, d=40), doing so moved mean calibration AUROC 0.5046 →
0.5234 and the false-pass rate at the 0.65 gate from 1.5% to 4.0%.

**The p0 negative control.** Percentile 0 is the output of `embed_tokens` at
the last prompt token. Under every PLAN §4/§6 prompt template that position is
a fixed template suffix, and a token embedding carries no position or context
— so p0 is the *same vector* for every question in a cell, and a probe on a
constant cannot separate anything. AUROC there must be ≈ 0.50. The pre-audit
tap scored **0.717**, which is how the audit established it was reading a
generated token rather than a prompt token. Making this control blocking means
that class of bug cannot reach a results table unnoticed again.

Layer selection happens on the **calibration split only**, never train or test
(PLAN §6·6, §14.2).

## Grading

| Field | Default | Notes |
|---|---|---|
| `NUMERIC_TOLERANCE` | `1e-6` | Relative, for GSM8K. |
| `USE_NLI_FALLBACK` | `True` | Local entailment for SimpleQA. Turning this off drops R3 to string matching and will likely fail Gate 1 for that family. |
| `NLI_MODEL` | `microsoft/deberta-large-mnli` | ~1.6 GB. **Always loaded fp32** — DeBERTa's disentangled attention has fp32-only kernels and throws under bf16. |
| `NLI_ENTAIL_THRESHOLD` | `0.70` | Bidirectional entailment required. |
| `NLI_BATCH_SIZE` | `64` | |
| `STRIP_ARTICLES` | `True` | Strip a/an/the in normalization. |

## Gates

| Field | Default | Notes |
|---|---|---|
| `GATE1_AGREEMENT` | `0.95` | Grading sanity (PLAN §16). |
| `GATE2_SPEARMAN` | `0.60` | Format agreement / H0. Applied to the bootstrap **lower CI bound**, not the point estimate. |
| `GATE4_REQUIRE_BASE_ELICITATION` | `True` | **Inert.** |
| `BASE_ELICITATION_MIN_PARSE_RATE` | `0.50` | **Inert** — inspect `t3_parse_and_accuracy` for the base model manually. |

## Statistics

| Field | Default | Notes |
|---|---|---|
| `N_BOOTSTRAP` | `2000` | Drop to ~200 for smoke runs; it dominates analysis wall-clock. |
| `BOOTSTRAP_CI` | `0.95` | |
| `ECE_BINS` | `15` | |
| `MURPHY_BINS` | `10` | Reliability/resolution/uncertainty binning. |
| `CALIBRATOR` | `"auto"` | `auto \| isotonic \| platt`. Auto picks isotonic when n ≥ `ISOTONIC_MIN_N`. |
| `ISOTONIC_MIN_N` | `200` | Below this, isotonic overfits — fall back to Platt (PLAN §8·1). |
| `MIN_DISTINCT_VERBAL` | `3` | Verbal pre-flight: a cell with fewer distinct confidence values is **excluded**, not reported as poorly calibrated (PLAN §8·6). |
| `GATE3_ENFORCE` (see Probe) | `True` | Also gates the **data path**: a cell failing Gate 3 contributes no `internal_cal` at all, and is listed in `derived/internal_gate3_skipped.json`. |
| `GROUND_TRUTH_VARIANT` | `"EXTRACT"` | Which generation defines `correct` for **all three** signals. Accuracy differs by up to 12 points across A/B/C/SAMPLE/EXTRACT, so scoring the verbal signal against one variant's correctness while the probe trains on another's compares three signals to three different targets (AUDIT finding 4). `EXTRACT` is the plain answering pass the activations were tapped from. |
| `QUADRANT_THRESHOLD` | `0.5` | Split point on calibrated scores for hopeful/suppressed. |
| `HLR_METHOD` | `"auto"` | `auto \| bayes_mixed \| cluster_robust`. Auto tries `BinomialBayesMixedGLM`, falls back to a cluster-robust logit. **Check `method` in the output before quoting coefficients.** |

## Checkpoint / IO

| Field | Default | Notes |
|---|---|---|
| `RESUME` | `True` | `False` **truncates** existing checkpoints. |
| `CHECKPOINT_EVERY` | `50` | Records between `fsync`ed flushes. |
| `SAVE_RAW_TEXT` | `True` | Keep full generations, not just parses. Needed to re-parse without regenerating, and for the judge. |
| `SAVE_ACTIVATIONS` | `True` | ~233 KB/question across all 5 models. 1.4 GB at N=1000. |
| `COMPRESS_ACTIVATIONS` | `True` | `npz` compressed shards. |
| `JSONL_ENSURE_ASCII` | `False` | |

## Figures

| Field | Default | Notes |
|---|---|---|
| `FIG_DPI` | `200` | |
| `FIG_FORMATS` | `("png", "pdf")` | PDF for LaTeX inclusion. |
| `FIG_WIDTH` | `7.2` | Inches — two-column figure width. |
| `LATEX_TABLES` | `True` | Emit `.tex` alongside every CSV. |
| `FIG_STYLE` | `"paper"` | **Inert** — only the paper style is implemented. |

---

## `JudgeConfig` (cell 18)

A separate object. Runs **last**, after every generation model is freed,
reading only saved JSON. Its agreement with the deterministic grader is the
reported Gate 1 statistic.

| Field | Default | Notes |
|---|---|---|
| `ENABLED` | `True` | Set `JudgeConfig(ENABLED=False)` to skip the audit. |
| `MODEL` | `Qwen/Qwen2.5-32B-Instruct` | ~65 GB bf16, loads **natively** — fits a 96 GB card with 35 GB to spare. |
| `FALLBACKS` | 14B, 7B | Tried in order if the primary fails to load. |
| `LOAD` | `"auto"` | `auto \| awq \| gptq \| bnb4 \| native`. See resolution rules below. |
| `TARGETS` | `("unresolved","fuzzy","audit")` | What to send. `audit` samples deterministically-graded items — that's the Gate 1 statistic. `all` is expensive. |
| `AUDIT_SAMPLE_PER_FAMILY` | `100` | Capped by availability. |
| `MAX_NEW_TOKENS` / `BATCH_SIZE` | `12` / `16` | The judge emits one `VERDICT:` line. |
| `FREE_AFTER` | `True` | Tear down after the audit. |

### Why the default is native, not quantized

A quantized checkpoint needs its **kernel package installed**, and those
packages lag new GPUs. On molab none are present, and AWQ ids fail with
`Loading an AWQ quantized model requires gptqmodel`. Since 96 GB fits a 32B in
bf16 outright, native is both simpler and stronger than a 14B-in-4-bit.

`LOAD="auto"` resolves against what is actually importable:

| Condition | Resolves to |
|---|---|
| id contains `awq` **and** `awq` importable | `awq` |
| id contains `gptq` **and** `gptqmodel`/`auto_gptq` importable | `gptq` |
| `bitsandbytes` importable | `bnb4` |
| otherwise | `native` (bf16) |

To use a 72B, install `gptqmodel` first and set
`MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"` (~41 GB). A bf16 72B is a 145 GB
download and will not fit 96 GB.

Quantization here does **not** violate PLAN §9.1 — that rule protects the
hidden states the probe reads, and the judge is never probed.

### Measured (smoke run, Qwen2.5-32B native)

276 items judged, **96.01% agreement — Gate 1 passes**.

| Grader | n | Agreement |
|---|---|---|
| numeric (GSM8K) | 49 | 100% |
| nli (SimpleQA) | 53 | 100% |
| alias_exact (PopQA) | 83 | 94.0% |
| symbolic (MATH) | 90 | 93.3% |

The two deterministic tiers disagree most — PopQA's alias list misses valid
surface forms, and sympy equivalence is stricter than semantic equivalence.
Those are the two families to prioritise in the manual check sheet.

---

## Inert fields

Declared but never read. Setting them does nothing:

`NOTES` · `STOP_ON_DOUBLE_NEWLINE` ·
`GATE4_REQUIRE_BASE_ELICITATION` · `BASE_ELICITATION_MIN_PARSE_RATE` ·
`FIG_STYLE`

`PROBE_C_GRID` was on this list; it is now consumed by `select_probe_C`.

They change `CFG.hash()`, so avoid editing them — you'd fork the fingerprint
without changing the run.

---

## Recipes

Cell 3 ends with `CFG = replace(_BASE_CFG, …) if SMOKE else _BASE_CFG`. Any
`CFG = …` you add **after** that line wins regardless of `SMOKE`, so the
`replace(_BASE_CFG, …)` recipes below bypass the smoke block entirely.

```python
# Smoke — every stage end-to-end in ~2 min (this is the shipped default)
SMOKE = True

# Full grid, one molab session (~7.1 GPU-hr)
SMOKE = False

# PLAN §3 target, two sessions
CFG = replace(_BASE_CFG, RUN_NAME="full", N_PER_CELL=2000,
              SKIP_MODELS=("qwen2.5-7b-instruct", "qwen2.5-7b-base"))
# ...then session 2, same RUN_NAME so it resumes the shared bank:
CFG = replace(_BASE_CFG, RUN_NAME="full", N_PER_CELL=2000,
              ONLY_MODELS=("qwen2.5-7b-instruct", "qwen2.5-7b-base"))

# Kaggle: small models only — 7B cannot finish there
CFG = replace(_BASE_CFG, RUN_NAME="kaggle", N_PER_CELL=500,
              SKIP_MODELS=("qwen2.5-7b-instruct", "qwen2.5-7b-base"),
              PURGE_WEIGHTS="after_run")

# Small models concurrently (helps: none of them saturate the GPU)
CFG = replace(_BASE_CFG, MODEL_EXEC="concurrent", MAX_CONCURRENT_MODELS=3,
              SKIP_MODELS=("qwen2.5-7b-instruct", "qwen2.5-7b-base"))

# Semantic Entropy Probes formulation instead of correctness labels
CFG = replace(_BASE_CFG, PROBE_LABEL="entropy")

# Re-run analysis only, on an existing run's raw output
CFG = replace(_BASE_CFG, RUN_NAME="full", STAGES=(
    "grade","entropy","probe","calibrate","stats","figures","tables","report"))

# Skip the judge audit (it is ON by default)
JUDGE = JudgeConfig(ENABLED=False)

# Bigger judge — only after `pip install gptqmodel`
JUDGE = replace(JUDGE, MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ", LOAD="awq")

# Judge everything, not just fuzzy + audit sample (expensive)
JUDGE = replace(JUDGE, TARGETS=("all",))
```
