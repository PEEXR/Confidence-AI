# Independent audit — `prod500`

**Scope:** branch `prod500`, commit `7570841` (320 files, +414,070 lines) against `2c5db4c`.
**Method:** artifacts re-derived from the committed files. Every number below was recomputed from
this repo, not read out of a summary. Where a claim is reproduced, it says so; where it is
contradicted, the contradicting file and key are named.
**Date:** 5 Sep 2026.

---

## 0. Verdict

**The run is real, and the previously-blocking defects are genuinely fixed.** After two audits in
which the headline rested on measurements that had not happened, this one holds up at the
measurement layer.

What remains is confined to the **reporting layer**: three of the five hypothesis verdicts in
`RESULT_SUMMARY.md` do not match the pipeline's own `t15_hypothesis_verdicts.csv`, and the single
supported hypothesis is stated at a precision its measurement grid cannot carry.

That is good news. Reporting defects cost weeks; measurement defects cost a re-run.

---

## 1. Verified fixes

Recomputed, not taken on trust:

| Prior finding | Status | Evidence |
|---|---|---|
| 30 cells "committed" from a 6 m 41 s CPU run with `cuda: false` | **Fixed** | `derived/compute_ledger.parquet` sums to **4.778 GPU-hours, 17,202 s, 81,575 generations, 14.96 M output tokens** across 420 stage-rows |
| **Gate 1 was `null`** while prose asserted "H2 supported with Gate 1 holding" | **Fixed; reproduces exactly** | Recomputed independently from `tables/gate1_manual_check_sheet.csv`: 200/200 rows filled, **195/200 = 97.50%** |
| 17 of 30 cells deleted below an accuracy floor | **Fixed** | `meta/config.json` → `COMMIT_CELLS_OUTSIDE_BAND: True`; all 30 committed, out-of-band accuracy kept as a covariate |
| Activation tap read generated tokens | **Fixed** | prefill-only tap; `audit_docs/FINDINGS.md` |
| No negative control | **Fixed — and it fired** | `P0_NEG_CONTROL_TOL: 0.1`, 20 label shuffles; `7b-base__C3` dropped at p₀ = 0.634 (`derived/internal_gate3_skipped.json`) |
| Headline built on never-measured cells | **A real guard now exists** | `derived/h3_model_delta.json` declines to test at `n_matched = 0` and records: *"The pre-repair code would have reported a rate of 0.0 here rather than declining to test."* |

Activation `.npz` sizes scale correctly with model size (0.5B → 3.3 MB … 7B → 13.4 MB), and raw
generations are present at every stage for all 30 cells. The run happened.

---

## 2. Blocking: the summary contradicts the pipeline

`RESULT_SUMMARY.md` and `tables/t15_hypothesis_verdicts.csv` were committed together and disagree.

| H | `t15_hypothesis_verdicts.csv` (generated) | `RESULT_SUMMARY.md` (written) |
|---|---|---|
| H0 | `passed=False` — falsified | FALSIFIED — *agrees* |
| **H1** | `passed=False` — "falsified — CIs overlap, no ordering distinguishable" | **"FALSIFIED on ECE; SUPPORTED on Resolution"** |
| **H2** | `passed=False` — "**not supported** — failing conjunct: `replicates_across_cells`… report as DESCRIPTIVE" | **"CONFIRMED ✓"** |
| H3 | untested | UNTESTED — *agrees* |
| H4 | `passed=True` | SUPPORTED — *agrees* |

### H2 is the serious one

`derived/h2_quadrants.json` contains `h2_pass = False` and
`conjuncts.replicates_across_cells = False`. Beneath it:

```
replication.has_year         n_cells_tested=2  n_cells_significant=0  replicates=False
replication.is_long          n_cells_tested=1  n_cells_significant=0  replicates=False
replication.has_multi_entity n_cells_tested=3  n_cells_significant=0  replicates=False
replication.has_number       n_cells_tested=3  n_cells_significant=0  replicates=False
replication.family           n_cells_tested=0  n_cells_significant=0  replicates=False

concentration.hopeful     present in 1 of 9 cells
concentration.suppressed  present in 2 of 9 cells
counts_labelled           Excessive Hedging = 90 · Performative Certainty = 9
```

**No feature reached significance in a single cell.** The phenomenon is present in one or two of
nine cells. The "10.8% vs 1.1% — LLMs are 10× more prone to hedging" headline rests on **90 items
versus 9**. The pipeline's instruction to report this descriptively is correct; the executive
summary promotes it to "CONFIRMED ✓".

H1 is the same move, milder: the generated verdict says no ordering is distinguishable; the summary
adds a "SUPPORTED on Resolution" half that appears in no generated artifact.

---

## 3. H4 is stated beyond the resolution of its grid

`meta/config.json` sets `PERCENTILES: [0, 25, 50, 75, 100]`. Probes are fit at five depths, so an
onset can take at most five values. Across all 22 present cells the observed values are
**{25, 50, 75} and nothing else.**

Recomputed from `tables/t8_h4_depth_onsets.csv`:

```
22 of 30 cells present
retrieval (n=11)  ten 25s, one 50              → mean 27.27%   (= 300/11)
reasoning (n=11)  six 25s, three 50s, two 75s  → mean 40.91%   (= 450/11)
delta                                            13.64%        (= 150/11)
```

The reported "27.3% / 40.9% / Δ = +13.64% / 95% CI [+2.27%, +27.27%]" are four-significant-figure
renderings of `300/11`, `450/11`, `150/11`, `25/11`, `300/11`. **The CI's lower bound is one cell
moving one grid step.**

Two checks run directly against the CSV:

- **Dropping the two C1 cells at 75%** (`3b-instruct__C1`, `7b-instruct__C1`) takes Δ from
  **13.64 → 6.06**. The effect is carried by two cells of the same *condition*, which is at least as
  consistent with a C1-specific effect as with a reasoning-vs-retrieval effect.
- **One-sided permutation test**, same 11 vs 11 onsets, 20,000 permutations: **p = 0.053.**

`RESULT_SUMMARY.md` states "strictly excludes 0" and "SUPPORTED ✓ (p < 0.05)". A percentile
bootstrap over a three-valued discrete grid is not reliable at that resolution, and a permutation
test on the identical numbers does not clear 0.05.

H4 is a **suggestive trend on 22 unbalanced cells at 25-percentage-point resolution**. Also,
`0.5b-instruct` contributes exactly one cell (C2), so "holds across model scale" is not supported by
this table.

**Highest-value remaining experiment in the repo:** re-run the probe sweep at `PERCENTILES` of every
5–10%. It is cheap against the 4.778 GPU-hours already spent and converts H4 from contested to
settled, in whichever direction.

---

## 4. Further findings

**4.1 Three inconsistent hardware records in one commit.**

| File | Platform | Device | Torch |
|---|---|---|---|
| `RESULT_SUMMARY.md` | Molab | NVIDIA RTX PRO 6000 Blackwell, 101.98 GB | — |
| `meta/run_log_rows.md` | molab | — | 2.11.0**+cu130** |
| `meta/provenance.json` | **local** | **AMD Radeon RX 6600, 8.57 GB** | **2.13.0** |

Also differing: `config_hash` (`b11fe302abde` vs `9a3828ff2949`), start time (01:35:57Z vs
05:43:45Z), `transformers` (5.14.1 vs 5.13.0). These are two executions. `compute_ledger.parquet`'s
4.778 GPU-hours matches `run_log_rows.md`, so the data came from the molab run — meaning
**`provenance.json`, the file a reproducibility checker opens first, describes a different machine**,
one that cannot hold Qwen2.5-7B in bf16 (~15 GB of weights against 8.57 GB VRAM).

**4.2 The run is not bound to code.** `meta/run_log_rows.md` states it directly:
`code_sha: UNBOUND-nogit` — *"UNBOUND — this run is not traceable to a commit"*. `provenance.json`
says `2c5db4c-dirty`. Neither binds 414 k lines of artifacts to reviewable source.

**4.3 Gate 1's disagreements are one-directional and tier-concentrated.** All five are
`automated=False, manual=True` — LaTeX brackets, `x = 60` prefixes, `11/15` vs `\frac{11}{15}`.
Four in C2, one in C3, **none in any R cell**. The automated grader systematically under-credits
math formatting, and the bias sits in one arm of the paper's central contrast. 97.5% is a sound
scalar and a misleading control: the residual 2.5% is tier-correlated, pointing the same direction
as H4 and every C-vs-R comparison. Fix: report agreement **per tier**, and re-grade C cells with a
formatting-normalising parser. This strengthens the result rather than weakening it.

**4.4 Gate 3's arithmetic does not close.** `RESULT_SUMMARY.md` says "22 of 28 cells passed". The
grid is 30; `internal_gate3_skipped.json` lists 6; 30 − 6 = 24. Neither 28 nor 22 is derivable from
the committed files.

**4.5 The regression behind the strongest prose claim is thinner than it reads.** "Verbal confidence
is actively deceptive" (β = −1.95) comes from `derived/hierarchical_regression.json`: **n = 1,166 ·
600 questions · 12 cells** (not 30), with **202 behavioral and 270 internal values imputed**
(17% / 23%, missingness indicators correctly included), and nearly every `per_cell` fit returning
`Singular matrix`. For fairness: H3's `n_matched = 0` is specifically the base-vs-instruct matched
comparison (701 excluded, 600 of them `7b-base`) — **not** a claim that no row has both signals. That
stronger reading was checked and is not supported. Still, a β from a 12-cell, 23%-imputed fit should
not carry the paper's boldest sentence.

---

## 5. Three things worth foregrounding

These are real and rare, and more persuasive to a reviewer than any current headline claim:

1. **The negative control fired and the cell was dropped** — `7b-base__C3`, p₀ = 0.634 against a
   pre-registered 0.50 ± 0.10.
2. **A bug that would have manufactured a result was found and disclosed in the artifact itself** —
   `h3_model_delta.json`.
3. **All 30 cells were committed**, out-of-band ones included, with accuracy retained as a covariate
   rather than used as a deletion criterion.

Unlike the current headlines, every one of these is true as written.
