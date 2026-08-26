# Audit repair tests

*(For what the repaired code actually found when run over the existing
data, see [`../FINDINGS.md`](../FINDINGS.md).)*

Each file targets specific findings in [`../AUDIT.md`](../AUDIT.md) and backs
the `[x]` marks in [`../AuditTasks.md`](../AuditTasks.md).

```
python audit_docs/tests/run_all.py
```

| File | Covers |
|---|---|
| `test_tap.py` | Finding 1. The tap captures the last **prompt** token, verified against hand-computed hidden states at all 5 depth percentiles under both left and right padding; the frozen buffer survives simulated decoding; p0 is provably constant across rows sharing a final token; an unrecognised architecture raises; a stale `MODEL_SPECS['layers']` raises in **both** directions rather than silently re-scaling every depth, and leaks no hooks when it does; the tap raises rather than mis-mapping depths. |
| `test_logprobs.py` | Finding 3. `sequence_logprobs` matches a naive per-token reference computed on real `transformers` `generate()` output — greedy, N=10 sampled, hot sampling with early stops, `max_new_tokens=1` — is invariant across chunk settings, and returns NaN rather than 0.0 for rows with no scored token. Skips cleanly if `transformers` is absent. |
| `test_entropy.py` | Finding 3. No false merges across ±∞ or large integers while genuine float-rounding spellings still merge; the NLI pass on math forms receives the question as context; `modal_share` stays count-based while `modal_mass` follows likelihood; Math-equivalent answers merge (`0.5 ≡ 1/2 ≡ \frac{1}{2} ≡ 2/4`, with `0.25` kept apart) with **and without** `math_verify` installed; NLI is reachable for math forms; union-find merging is transitive; log-prob weighting activates and falls back cleanly; `n=1` yields NaN rather than confidence 1.0. |
| `test_analysis.py` | Findings 2, 4, 6, 7, 8, 9. The entropy probe label's median is computed on train rows only rather than across splits; the GLMM records imputed counts and adds a missingness indicator rather than silently mean-filling missing-not-at-random values; H2 records how many rows its composite axis was built from; Format B is never canonical even when constructed to have the lowest Brier; H3 declines instead of reporting a rate built from NaN coercion; H2 is blocked by a null Gate 1 and names its failing conjuncts; the GLMM declares four random intercepts; Gate 3 fails on the audit's exact p0 = 0.717 case and on a missing p0; an unfilled Gate 1 sheet reads as `None`; popularity and difficulty-matching outputs are well-formed. |
| `harness.py` | Loads the pipeline's pure functions with `PATHS` / `LOG` / `json_write` stubbed, so the analysis code can be exercised without a GPU or a run directory. |

## These tests are mutation-tested

Adversarial verification caught the suite passing while the code was wrong —
twice. A guard that greps the source for `correction=False` is satisfied by a
*comment* containing that string; a behavioural check on `internal_scores`
passed because the fixture never reached the code path it meant to exercise.

So the guards are now checked by deliberately reintroducing each defect and
confirming the suite fails. Thirty-eight mutations are caught, including: reverting
the observed χ² to Yates-corrected; Yates-correcting the permutation null
instead; `internal_scores` ignoring its Gate 3 verdict; `gate3=None` failing
open; moving probe-`C` selection back onto the calibration split; the parser
dividing by 100 unconditionally; the ratio branch dropping its denominator
bound (`50/50` → 1.0); replication keying off the uncorrected p; the GLMM
claiming random intercepts a cluster-robust fallback does not have; the H2
conjunct reverting to a grid-wide `any()`; H4 ignoring Gate 3; the GLMM
losing its all-NaN-column guard; Format B becoming an eligible canonical format
again; the ratio branch falling through instead of returning None; the
percentage regex losing its lookbehind (so `"90-95%"` reads as -95); the
per-cell GLMM formula being hardcoded past the surviving signals; `gate3_filter_sweep`
failing open; and every keyword argument in `run_pipeline` that carries a gate
to the function enforcing it.

That last group needs its own note. Every gate is enforced inside a function,
and every one can be reverted by deleting a keyword at the single place it is
called. `run_pipeline` needs a GPU and a run directory, so no test can execute
it — which is exactly why a wiring mistake there is invisible. The suite walks
its AST instead, asserting that `test_h4_depth` receives `gate3`,
`test_h2_quadrants` receives `gate1_pass` and `gate3`, `assemble_signals`
receives `gate3`, `stage_figures` receives the *filtered* sweep, and
`stage_judge` runs before the stats block. Structural, but the structure is
the property being asserted — these are claims about the call graph, not greps
for a string.

Three of those were found only after a verification pass mutation-tested the
suite itself and reported four survivors. A fifth was found because the
end-to-end block had been appended *below* `sys.exit()` and had never run at
all — which is its own lesson: check that a new test can fail before trusting
that it passes.

Two consequences for anyone editing these tests:

- **Assert on behaviour, not on source text.** Where a structural check is
  genuinely the right tool, walk the AST (`params_of`, `_chi2_calls`) rather
  than grepping a source segment.
- **Check the fixture can fail.** Several tests assert that the negative case
  produces zero rows; each is paired with a positive control confirming the
  same call produces rows when it should. A 2×2 contingency fixture exists
  specifically because Yates' correction applies to no other shape, and the
  default 4×2 fixture cannot detect that class of bug at all.

`harness.py` deliberately does **not** import `confidence_pipeline` — the
module has heavy import-time side effects (dependency installation, path
setup, and it runs the pipeline at the bottom). It parses the source and
`exec`s the functions it needs instead, which also means the tests read the
same source the notebook is generated from.

Tests write scratch files to `_out/`, which is safe to delete.
