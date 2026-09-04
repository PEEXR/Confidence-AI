# When Verbalized Confidence Isn't There — pipeline

**Elicitation degeneracy, format disagreement, and what survives: a
measurement-validity study of confidence in small LLMs.**

Single-notebook implementation of [`PLAN.md`](PLAN.md) v3: measure LLM
confidence three independent ways on the same questions, and study where they
**disagree**.

| Signal | What it reads | Cost per question |
|---|---|---|
| **Verbalized** | what the model *says* about its certainty (formats A/B/C) | 3 generations |
| **Behavioral** | semantic entropy over N=10 resampled answers | 10 generations |
| **Internal** | logistic probe on hidden states, 5 depth percentiles | 1 forward pass |

## Headline result

**The verbalized-confidence axis is largely not measurable in this grid, and the
analyses that appeared to find effects in it were measuring its absence.**

Re-running the analysis stages over the existing run with the post-audit code
(`python audit_docs/reanalyse.py`, no GPU) gives:

| | |
|---|---|
| Cells where verbalized confidence is degenerate | **24 of 30** |
| Format agreement (Gate 2) | **Falsified** — A–C ρ = &minus;0.047 |
| The one well-calibrated format (B) | **Circular** — ECE = 0.0000 exactly |
| H3 &mdash; base vs instruct | **Untestable** — 0 matched rows carry both signals |
| H2 &mdash; quadrant taxonomy | **Not supported** — all 111 cases from 2 of 6 cells |
| Behavioural entropy vs stated confidence | **4.3&times; the resolution**, full grid coverage |

Nothing below 3B produces a usable verbal signal on any tier, and where it
survives the model uses three or four distinct values on a 0&ndash;100 scale. The
three elicitation formats do not correlate with each other. See
[`audit_docs/FINDINGS.md`](audit_docs/FINDINGS.md) for the full tables.

**One ablation could overturn this.** Three or four distinct values may mean the
prompt is the binding constraint rather than the model. A coarser scale (0&ndash;10)
or a logit-based readout at 1.5B would settle whether §1 is a finding about these
models or about this elicitation. That should run before anything is written up.

## Unmeasured: the internal signal

The depth-of-decodability result &mdash; correctness linearly decodable at roughly
25% depth, retrieval ceilings 0.84&ndash;0.91 against 0.51&ndash;0.80 for reasoning &mdash;
was the most robust finding the pipeline produced, and it is currently
**unmeasured**, not merely stale:

- it was computed from activations captured by the broken tap, which read the
  last *generated* token rather than the last *prompt* token; and
- `PROBE_LABEL` now defaults to `"entropy"` rather than `"correct"`, so the probe
  answers a different question than the one that produced those numbers.

The activation shards are not in the repository, so this cannot be recomputed
without re-running the grid. Treat the figures above as a hypothesis.

## The 2&times;2, retired as a claim

The epistemic-alignment quadrants are still computed and still reported:

| | Behavioral low | Behavioral high |
|---|---|---|
| **Verbal high** | **Performative Certainty** &mdash; stated confidence exceeds sample stability | **Grounded Certainty** |
| **Verbal low** | **Honest Doubt** | **Excessive Hedging** &mdash; stated confidence under-reports unanimous consensus |

But they are no longer offered as a finding. Four of the six contributing cells
never cross the 0.5 threshold at all, so they are structurally incapable of
producing half the taxonomy, and every Performative Certainty case comes from two
cells. The earlier "hopeful confidence" framing has been retired &mdash; it
anthropomorphizes, and the result carrying it was cell-determined.

## Audit status

An external audit (`audit_docs/AUDIT.md`) found four blocking defects. All are
fixed in the code and covered by a mutation-tested suite. The trees in
`results*/` predate the fixes; `results_repaired/` holds the analysis stages
re-run over the existing generations with the corrected code.

| Defect | Fix | Results regenerated? |
|---|---|---|
| Activation tap read the last **generated** token, not the last prompt token | Dedicated prefill pass, tap frozen before decode; p0 negative control now blocks Gate 3 | ❌ needs a GPU run |
| Format B's "confidence" was the empirical accuracy of the chosen bucket | B can never be canonical; label-free `Bfix` mapping added; B kept as a supervised reference | ✅ **confirmed** — ECE = 0.0000 exactly |
| Semantic entropy used raw sample counts; NLI merging skipped math | Sequence log-probs, length-normalised; symbolic equivalence merging; entropy normalised by `N_SAMPLES` | ⚠️ partly — normalisation re-run; log-prob weighting needs a GPU run |
| H3 / Figure 3 coerced NaN to 0.0 | Indicator built only over rows with both signals present; H3 declines rather than reporting a rate built from missing data | ✅ **confirmed** — 0 testable rows |

**Withdrawn outright, not pending re-measurement: H2 and H3.** The re-analysis
shows H3 has no testable rows and H2's cases come from two cells. These are not
weak effects awaiting more data; the axis they were measured on is largely
absent. See [`audit_docs/FINDINGS.md`](audit_docs/FINDINGS.md).

The verbal half of the analysis needs no GPU and is reproducible now:

```bash
python audit_docs/reanalyse.py          # -> results_repaired/
```

It re-runs the verbal axis, the entropy normalisation and H0–H3 over
`results/derived/graded.parquet`, which survived the runs. Signal 3 and the
log-prob weighting need a full production run — see *Quick start*. Set
`CODE_SHA` in the environment first, or the run will not be traceable to a
commit.

## Related work

The bare finding that verbalized confidence runs hotter than internal signals
was published four separate times in 2026, so this pipeline is positioned as a
**measurement-validity / negative-results** contribution rather than a
discovery.

- **DECK: A Consistency × Confidence Taxonomy of LLM Hallucinations**
  ([arXiv:2606.02289](https://arxiv.org/abs/2606.02289), Jun 2026) — a direct
  structural twin of the quadrant taxonomy: inter-sample consistency ×
  confidence, four named regimes, 3 models × 4 datasets.
- **Mahaut et al., ACL 2024** — already compares verbalized confidence,
  sequence probability, P(True), and hidden-state probes on shared items.

The re-analysis sharpens the positioning rather than softening it. A paper that
reports *why* a cross-signal comparison of confidence fails — a circular format,
three elicitations that do not correlate, an axis that is degenerate below 3B —
is a stronger contribution than one that reports a disagreement measured on that
axis.

**What is also new here is the base-rate control.** Every published
cross-signal comparison confounds method signal with item difficulty. This
pipeline holds difficulty fixed by matching within cells and by a multi-level
model with cell random intercepts, rather than by deleting cells — see
`difficulty_matched.json` and `hierarchical_regression.json`.

- **`confidence_pipeline.py` — the source of truth.** Git-diffable, split into
  `# %%` cells.
- `confidence_pipeline.ipynb` — the runnable deliverable, **generated from the
  `.py`**. Do not edit it by hand; edits there are overwritten.
- **[`CONFIG.md`](CONFIG.md) — every knob, what it does, and what it costs.**
- `audit_docs/` — the external audit, the repair checklist, and the tests
  backing it (`python audit_docs/tests/run_all.py`).

### Keeping the two in sync

```bash
python audit_docs/sync_notebook.py     # .py  ->  .ipynb
```

**Run this after every edit to the `.py`.** Nothing enforces it automatically,
and the failure is silent and expensive: the notebook is what actually executes
on molab/Kaggle, so an unsynced repo runs the *old* code while the diff shows
the new. The converter is verified lossless — regenerating from the pre-repair
`.py` reproduced the pre-repair notebook byte-for-byte — and is idempotent.

---

## Quick start

### molab (recommended)

1. Open a notebook, click the specs button in the header, attach the GPU.
2. Upload `confidence_pipeline.ipynb` (or `confidence_pipeline.py` if you
   prefer marimo). Run `python audit_docs/sync_notebook.py` first if you have
   touched the `.py`.
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

**6 tiers × 5 model variants = 30 cells.** Under the post-audit design,
`COMMIT_CELLS_OUTSIDE_BAND = True` defaults to committing all cells. Rather than
deleting cells outside the 25–80% pilot accuracy band (which previously eliminated all of
R2, R3, and 0.5B), base-rate difficulty is controlled statistically via cell random
intercepts in the multi-level GLMM (`in_band` / `pilot_accuracy` covariates) and
within-tier difficulty matching — see `difficulty_matched.json` and
`t2_cell_commitment.csv`.

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
to `meta`, and calls `gc.collect()` + `empty_cache()` + `ipc_collect()`.

Disk is separate: `PURGE_WEIGHTS` is `never` / `after_model` / `after_run`.
Note that each model is loaded **twice** — once for the pilot pass, then again
for the real generation stages, because the band gate needs every model's
pilot before any cell can be committed. The purge therefore fires only on a
model's final pass; `after_run` clears the whole cache once at the end.

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
- Probe labels now default to `PROBE_LABEL="entropy"` (PLAN §6·5), so the
  probe tests whether prompt-boundary activations predict multi-sample
  dispersion. `PROBE_LABEL="correct"` is still available as a documented
  supervised reference — but with it, two of the three "independent" signals
  are supervised accuracy predictors and the comparison is not symmetric.
  When `correct` is used, labels come from the `EXTRACT` variant (the greedy
  pass whose activations were tapped); if that stage was skipped it falls
  back to `FORCED` and then to a `SAMPLE` draw, logging
  `probe_label_fallback` — a `SAMPLE` label is noise relative to its paired
  activation.
- **The 25–80% accuracy band no longer deletes cells.** It is recorded as an
  `in_band` covariate and the base-rate control is applied statistically.
  Set `COMMIT_CELLS_OUTSIDE_BAND=False` to restore the old behaviour, which
  removed R2, R3, 0.5B and 5/6 of C3 on the strength of an n=100 pilot.
- **Difficulty matching needs difficulty variation.** `difficulty_matched_view`
  bins questions by their cross-model correctness rate within a tier. A tier
  where every model scores near 0% (or near 100%) has no variation to bin on —
  which is exactly the regime the 25-80% band used to delete. Those tiers are
  named in `difficulty_matched.json` under `degenerate_tiers`, and for them the
  base-rate adjustment rests entirely on the GLMM's model × tier random
  intercept. Do not report a matched comparison that did not happen.
- **Grid coverage is the main outstanding gap.** PLAN targets N=2000 per
  cell; no cell has reached it, and 13/30 cells carry real data. R2, R3 and
  the 0.5B row now execute, but until they finish, any size-axis claim
  should be dropped rather than reported from a single tier.
- **Gate 1 closes on the human check sheet, not the judge.** A `null` verdict
  in `final_report.json` means unverified — it must not be read as passed.
  Only `results_v2.5/tables/gate1_manual_check_sheet.csv` is filled (200/200
  rows, 97.5% agreement); the sheets in `results_v2_flawed/` and
  `results_prerepair/` are blank. Filled sheets are no longer overwritten by
  a re-run.
- The judge needs a quantization kernel package for any AWQ/GPTQ id. Without
  one it falls back to native bf16, which is why the default is a 32B rather
  than a 72B.

---

## Reproducibility

Every derived artefact carries seed, config hash, code SHA, model revision,
platform, and dtype (PLAN §14.4 / X1). `CFG.hash()` is a 12-char SHA over the
full config — two runs with the same hash used the same knobs.

`code_sha` resolves from `$CODE_SHA`, then by walking up from the source file
for a `.git` directory, and marks the tree `-dirty` when it is. **On
molab/Kaggle the notebook is uploaded without its repository**, so set the
environment variable before running:

```python
import os; os.environ["CODE_SHA"] = "<output of git rev-parse --short HEAD>"
```

Without it the run records `UNBOUND-nogit`, emits a `RuntimeWarning`, and the
generated `run_log_rows.md` flags itself as untraceable. Every earlier run
recorded `nogit`, which is why none of them is bound to a code version.
After each production run, paste `meta/run_log_rows.md` into PLAN.md §17.2 and
update §0.
