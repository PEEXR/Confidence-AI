# Genuine vs. Hopeful Confidence in LLMs — pipeline

Single-notebook implementation of [`PLAN.md`](PLAN.md) v3: measure LLM
confidence three independent ways on the same questions, and study where they
**disagree**.

| Signal | What it reads | Cost per question |
|---|---|---|
| **Verbalized** | what the model *says* about its certainty (formats A/B/C) | 3 generations |
| **Behavioral** | semantic entropy over N=10 resampled answers | 10 generations |
| **Internal** | logistic probe on hidden states, 5 depth percentiles | 1 forward pass |

The headline output is the **quadrant analysis**: questions where verbalized
confidence is high but behavioral/internal say the model is guessing
("hopeful" / performed confidence), versus the reverse ("suppressed").

- `confidence_pipeline.ipynb` — the deliverable. Runs on molab or Kaggle.
- `confidence_pipeline.py` — same code as `# %%` cells; the git-diffable
  source and the `marimo convert` input.
- **[`CONFIG.md`](CONFIG.md) — every knob, what it does, and what it costs.**

---

## Quick start

### molab (recommended)

1. Open a notebook, click the specs button in the header, attach the GPU.
2. Upload `confidence_pipeline.py`, or convert first:
   `marimo convert confidence_pipeline.ipynb -o confidence_pipeline.py`
3. Run all cells. Cell 3 ships with `SMOKE = True` — a ~2 minute end-to-end
   validation on 5 questions/cell.
4. Set `SMOKE = False` for the real run.

### Kaggle

Upload `confidence_pipeline.ipynb`, enable GPU + internet, run all. Paths and
dtype auto-detect. **Read the compute ceiling below before planning a run** —
Kaggle cannot finish the full grid.

---

## Hardware reality

Measured on molab (RTX PRO 6000 Blackwell, 96 GB, sm_120, native bf16).

| N per cell | molab, all 5 models | Kaggle 2×T4, 7B pair only |
|---|---|---|
| 100 (pilot) | 0.7 GPU-hr | 23.8 GPU-hr |
| 1000 (default) | **7.1 GPU-hr** | 238 GPU-hr |
| 2000 (PLAN §3 target) | 14.2 GPU-hr | 476 GPU-hr |

Two consequences worth stating plainly:

1. **PLAN §10's compute note is off by roughly an order of magnitude.** It
   omits the ~14.2 generations per question (3 formats + forced-answer + N=10
   sampling + greedy extraction) and the CoT token budgets on GSM8K/MATH.
   Against a 30 GPU-hr/**week** cap, Kaggle cannot finish the two 7B variants
   even at pilot size. Use molab for anything involving 7B.
2. **Blackwell's native bf16 retires standing risk #1** (T4 FP16 non-finite
   activations, PLAN §16). Measured `nonfinite_frac = 0.0` across all
   activation shards. The Gate 3 finiteness pre-check still runs — you don't
   drop a pre-registered control — but it now passes trivially. That belongs
   in §00 as *"retired by hardware change, not by measurement."*

Qwen2.5-7B is 15.2 GB in bf16 and **will not fit a single 16 GB T4**. On
molab it fits with 80 GB to spare; on Kaggle it needs `device_map="auto"`
sharding across both cards, which pipeline-parallelises and idles half the
compute.

---

## The grid

**6 tiers × 5 model variants = 30 cells.** A cell is committed only after its
pilot lands in the 25–80% accuracy band. A ragged grid is the *planned*
outcome (PLAN §3, §10), not a failure — see `t2_cell_commitment.csv`.

| Tier | Source | Family | Answer form | Grader |
|---|---|---|---|---|
| R1 | PopQA, top popularity quintile | retrieval | entity | alias list |
| R2 | PopQA, bottom quintile | retrieval | entity | alias list |
| R3 | SimpleQA | retrieval, adversarial | short | local NLI |
| C1 | GSM8K | reasoning | numeric | exact numeric |
| C2 | MATH levels 1–2 | reasoning | LaTeX | sympy / `math_verify` |
| C3 | MATH levels 4–5 | reasoning | LaTeX | sympy / `math_verify` |

Models: Qwen2.5 `0.5B / 1.5B / 3B / 7B`-Instruct (the scaling ladder) plus
`7B-base` (the H3 post-training contrast).

Questions are a **random sample at a fixed seed** — never the head of a sorted
dataset — split 60/20/20 train/calibration/test inside each cell. The pilot
and agreement subsets are drawn from *train only*, so the band gate never
touches calibration or test (PLAN §14.2).

---

## No LLM judge

Grading is deterministic by construction. Every graded row records **which
tier resolved it**, so the audit trail is inspectable (`t4_grader_tier_usage`).

```
exact / alias  →  numeric  →  symbolic  →  local NLI  →  unresolved
```

Measured on the smoke run: `alias_exact 83 · symbolic 90 · numeric 49 ·
nli 53 · string_fallback 1 · unresolved 0`.

MCQ was considered and rejected: it collapses semantic entropy to a 4-bin
histogram, replaces the difficulty band with a 25% chance floor, and requires
fabricating distractors — a new confound in exactly the retrieval-vs-reasoning
contrast H4 tests.

Models answer in **rigid `KEY: VALUE` lines**, not JSON — far higher
compliance at 0.5B and on the base model. Responses are *stored* as JSON.
Parse failure is measured, never raised: `parse_ok` is a per-cell reportable
degeneracy, not an exception.

A **post-hoc judge** (cell 18) loads Qwen2.5-32B *after* every generation model
is freed, reads only the saved JSON, and produces a second opinion. Its
agreement with the deterministic grader is the reported Gate 1 statistic. It
does **not** replace the manual check sheet.

Measured on the smoke run — 276 items, **96.01% agreement, Gate 1 passes**:

| Grader | n | Agreement |
|---|---|---|
| numeric (GSM8K) | 49 | 100% |
| nli (SimpleQA) | 53 | 100% |
| alias_exact (PopQA) | 83 | 94.0% |
| symbolic (MATH) | 90 | 93.3% |

The deterministic tiers are where the disagreement lives: PopQA's alias list
misses valid surface forms, and sympy equivalence is stricter than semantic
equivalence. Prioritise those two families when hand-filling the check sheet.

The judge runs **natively in bf16, not quantized** — no quantization kernel
package is installed on molab, and 96 GB fits a 32B outright. See CONFIG.md
§ JudgeConfig for the `LOAD="auto"` resolution rules and how to use a 72B-AWQ
instead. Quantizing the judge would be fine in principle: PLAN §9.1's
clean-activation rule protects *probed* models, and the judge is never probed.

---

## Notebook layout

| Cell | Contents |
|---|---|
| 1 | bootstrap, imports, platform + device detection |
| 2 | `TIER_SPECS` / `MODEL_SPECS` registries |
| **3** | **`CFG` — every knob. Nothing below needs editing.** |
| 4 | paths, provenance (X1), `Checkpoint`, `RunLog` |
| 5 | question-bank builder + splits |
| 6–7 | prompts (A/B/C/FORCED/SAMPLE) and `KEY: VALUE` parsers |
| 8 | graders; 8b: pre-flight compute estimate |
| 9–11 | model manager, activation hooks, generation engine |
| 12–13 | stage drivers, band gate, grading, semantic entropy |
| 14–15 | probe sweep + Gate 3; calibration and scoring rules |
| 16–17 | signal assembly; H0–H4, quadrants, Omniscience-Index, HLR |
| 18 | post-hoc judge + Gate 1 manual check sheet |
| 19–21 | figures, table export, measured compute ledger |
| 22–23 | `run_pipeline()` and RUN |

---

## Outputs

Everything lands under `<OUTPUT_ROOT>/<RUN_NAME>/`:

```
data/         question_bank.json + manifest
raw/          {stage}/{model}/{tier}.jsonl   <- every generation, resumable
activations/  {model}__{tier}.npz            <- 5 percentile taps, float32
              + .finiteness.json             <- Gate 3 pre-check
derived/      graded · entropy · probe_sweep · signals · quadrants
              h0_gate2 · h1_calibration · h2_quadrants · h3_model_delta
              h4_depth · gate3 · bucket_mapping · compute_ledger
figures/      fig1_calibration · fig2_quadrant · fig3_model_delta
              fig4_depth_prediction · fig5_cell_commitment_grid
              fig6_signal_correlations      (png + pdf + .caption.txt)
tables/       t1..t15 (csv + parquet + tex) + gate1_manual_check_sheet.csv
meta/         provenance · config · final_report · run_log_rows.md
logs/         events.jsonl
```

Figures carry a caption naming their `predicted under H#` and the null they'd
show if falsified. Tables export to LaTeX for direct paper inclusion.
`meta/run_log_rows.md` is pre-formatted for PLAN.md §17.2.

---

## Checkpointing and resume

Every stage is **idempotent and resumable**, keyed by `(qid, variant)`, with
`fsync` on flush. Re-running a completed stage costs one file read.

This is what makes the notebook safe under marimo's reactivity: a reactive
re-run re-reads checkpoints and skips completed work rather than regenerating.
It's also what lets a run span sessions — set `ONLY_MODELS` to a subset,
run, then run the rest; the question bank and checkpoints are shared.

After **every** model finishes, the pipeline drops references, moves the model
to `meta`, and calls `gc.collect()` + `empty_cache()` + `ipc_collect()`, then
optionally deletes the HF snapshot (`PURGE_WEIGHTS_AFTER_MODEL`).

---

## Gates

| Gate | Fires | Pass rule | Where |
|---|---|---|---|
| 1 — grading sanity | before the full pipeline | ≥95% agreement with manual check | `judge_agreement.json` + manual sheet |
| 2 — format agreement | after the §4 subset | all pairwise Spearman lower CI ≥ 0.6 | `h0_gate2.json` |
| 3 — probe validity | before trusting §6 | ≥1 percentile AUROC ≥ 0.65, beats shuffle null **and** surface baseline; activations finite | `gate3.json` |
| 4 — model comparison | before the H3 claim | delta CI excludes 0 + missed-knowledge guard | `h3_model_delta.json` |

Gate 1 is **not** closed by the automated judge alone. Fill in
`tables/gate1_manual_check_sheet.csv` by hand — the judge is a second
automated opinion, not a human.

---

## Known limitations

- `MODEL_EXEC="resident"` currently plans the same waves as `"sequential"`;
  the distinction is not yet wired through.
- `MODEL_REPLICAS > 1` shards tiers across replicas of the same weights. It
  is **usually the wrong lever** — batched decode is memory-bandwidth bound,
  so replicas re-read their own weight copies. Raise `BATCH_SIZE` instead.
- Six config fields are declared but not consumed. See CONFIG.md § Inert.
- The hierarchical regression falls back to a cluster-robust logit when
  `BinomialBayesMixedGLM` fails to converge; check `method` in
  `hierarchical_regression.json` before quoting coefficients.
- **At small `N_PER_CELL` the probe stage skips.** It requires ≥20 train and
  ≥10 calibration rows per cell; the smoke default (`N_PER_CELL=5` → 3/1/1
  after the 60/20/20 split) is far below that, so every cell logs
  `probe_skipped` and Gate 3, H1 and H4 come back empty. That is a guard, not
  a bug — a probe fitted on 3 points validated on 1 would report noise as
  signal. Set `SMOKE = False` and all of it populates.
- The judge needs a quantization kernel package for any AWQ/GPTQ id. Without
  one it falls back to native bf16, which is why the default is a 32B rather
  than a 72B.

---

## Reproducibility

Every derived artefact carries seed, config hash, code SHA, model revision,
platform, and dtype (PLAN §14.4 / X1). `CFG.hash()` is a 12-char SHA over the
full config — two runs with the same hash used the same knobs.
