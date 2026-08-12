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
| `extract` | Signal 3, 5-percentile hooks at last prompt token | §6 |
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
| `N_SAMPLES` | `10` | PLAN §5 N=10. The dominant cost term. |
| `N_FEWSHOT_BASE` | `4` | Exemplars for `7b-base`, which has no chat template. |
| `STOP_ON_DOUBLE_NEWLINE` | `False` | **Inert.** |

## Execution policy

| Field | Default | Notes |
|---|---|---|
| `MODEL_EXEC` | `"sequential"` | `sequential \| resident \| concurrent`. `resident` currently plans identically to `sequential`. |
| `MAX_CONCURRENT_MODELS` | `2` | Only used under `concurrent`. |
| `CONCURRENT_MAX_PARAMS_B` | `4.0` | Models above this **always run alone**. |
| `MODEL_REPLICAS` | `1` | Replicas of the same weights, each taking a disjoint tier slice. |
| `PURGE_WEIGHTS_AFTER_MODEL` | `False` | Delete the HF snapshot after a model finishes. Turn **on** where storage is tight — all five models are 40.7 GB. |
| `EMPTY_CACHE_EVERY_BATCHES` | `4` | `torch.cuda.empty_cache()` cadence during generation. |

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
| `ACCURACY_BAND` | `(0.25, 0.80)` | PLAN §3. Cells outside are excluded — a ragged grid is the planned outcome. |
| `COMMIT_CELLS_OUTSIDE_BAND` | `False` | `True` bypasses the gate and **records the violation**. Needed for smoke runs, where a 5-question pilot can't land in the band meaningfully. |

## Probe (Signal 3)

| Field | Default | Notes |
|---|---|---|
| `PERCENTILES` | `(0, 25, 50, 75, 100)` | Depth taps. `0` = embedding output, `100` = final block. |
| `PROBE_LABEL` | `"correct"` | `correct` = ground truth; `entropy` = median-split semantic entropy (the Semantic Entropy Probes formulation, PLAN §6·5). |
| `PROBE_MAX_ITER` | `2000` | Logistic regression iterations. |
| `PROBE_STORE_DTYPE` | `"float32"` | PLAN §16 standing risk 1 — never store activations in half precision. |
| `AUROC_GATE` | `0.65` | Gate 3 threshold and the H4 onset definition. |
| `LABEL_SHUFFLE_REPEATS` | `20` | Null distribution size. Winner must beat its p95. |
| `SURFACE_BASELINE` | `True` | TF-IDF prompt-only control (PLAN §14.1). **Leave on** — §17.3 requires every activation measurement to ship one. |
| `PROBE_C_GRID` | `(0.01, 0.1, 1.0, 10.0)` | **Inert** — the probe uses `C=1.0` fixed. |

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

`NOTES` · `STOP_ON_DOUBLE_NEWLINE` · `PROBE_C_GRID` ·
`GATE4_REQUIRE_BASE_ELICITATION` · `BASE_ELICITATION_MIN_PARSE_RATE` ·
`FIG_STYLE`

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
              PURGE_WEIGHTS_AFTER_MODEL=True)

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
