# Decoding My Research Pipeline — Full Notes (Cells 1–23)

> Personal reference while reading through the LLM confidence-calibration
> codebase (Qwen2.5, three confidence signals: verbal / behavioral /
> internal). Written to be readable without re-opening the code — you
> should be able to explain any cell to someone else after reading its
> section.

**One-line summary of the whole notebook:** *pick 5 Qwen2.5 models × 6
question categories → make each model answer + state confidence three
different ways (talking, repeating itself, internal brain-scan) → grade
everything → check if the three "confidences" agree with reality and with
each other → turn the answer into a paper.*

---

## Glossary (terms I keep forgetting)

| Term | Plain-English meaning |
|---|---|
| **Weights** | The downloaded model file itself (the "brain"). 1GB–14GB depending on model size. |
| **Dataclass** | A Python shortcut for a class that's basically a bundle of settings. |
| **VRAM** | GPU memory — separate from normal RAM/disk. |
| **Temperature** | Controls randomness when generating text. `0` = always the single most likely answer. Higher = more varied. |
| **Greedy decoding** | Temperature = 0. Deterministic — always the top-probability token. |
| **Sampling** | Temperature > 0. Model can pick less-likely tokens — gives variety across repeated runs. |
| **Top-p** | A second randomness guardrail: only sample from the smallest set of tokens whose probabilities add up to p, so sampling can't produce total nonsense. |
| **Few-shot** | Showing the model a few worked examples before the real question, so it learns the expected format. Needed only for base (non-chat) models. |
| **Hidden state / activation** | A snapshot of the model's internal numbers mid-processing — "peeking at its thinking" before it's finished answering. |
| **Probe** | A small classifier trained on those internal snapshots to see if they secretly predict something (e.g. "is the model about to be right?"). |
| **AUROC** | Score from 0.5 (random guessing) to 1.0 (perfect) measuring how well something predicts an outcome. |
| **Label shuffle / null baseline** | Scrambling right/wrong labels on purpose to see how well a probe could "predict" by pure chance — the real probe is judged against this. |
| **TF-IDF / surface baseline** | An old-school text-pattern-matching method (no real "understanding"), used as a control. If it predicts correctness as well as the fancy probe, the probe isn't adding much. |
| **NLI (Natural Language Inference)** | A model trained to judge whether two sentences mean the same thing, even worded differently. Used as a backup grader here. |
| **Calibration** | Making a confidence score "honest" — e.g. things called "80% confident" really are right ~80% of the time. |
| **ECE / Brier score** | Numbers measuring how "honest" a confidence score is overall. |
| **Murphy decomposition** | Splits a calibration score into separate parts (how honest / how discriminating / how hard the task is) instead of one blended number. |
| **Spearman correlation** | How much two things agree in *ranking*, not exact values. |
| **Bootstrap / confidence interval (CI)** | Resample your data many times to estimate how uncertain a result is, then report a range you're fairly sure contains the truth. |
| **Checkpointing** | Regularly saving progress so a long job can resume instead of restarting from zero after a crash. |
| **Provenance** | A "receipt" recording exactly what settings/code/versions produced a result, so it's traceable later. |
| **Seed** | A starting number for randomness generators. Lock it → "random" results become reproducible. |
| **JSONL** | A text file where every line is its own JSON record — easy to append to and resume. |
| **Git SHA** | A short fingerprint of the exact code version that produced a result. |
| **Forward hook** | A small function PyTorch runs automatically every time a specific layer finishes processing — used here to "tap" internal numbers without modifying the model. |
| **KV cache** | Memory the model keeps around while generating so it doesn't recompute earlier tokens — the main thing that eats VRAM during generation. |
| **Union-Find** | A classic fast data structure for grouping items into clusters ("is A in the same group as B?") — used here to merge semantically-equal answers. |

---

# PART 1 — Bootstrap and Config (Cells 1–4)

## Cell 1 — Bootstrap: installs, imports, "where am I running?"

**Big picture:** the very first thing the notebook does is figure out its
own environment — nothing scientific happens here, it's just making sure
the ground is solid before anything else runs.

- **Auto-install** — checks a list of required packages (`torch`,
  `transformers`, `datasets`, `sklearn`, etc.) and pip-installs anything
  missing. `math_verify` (a robust math-answer checker) is installed too,
  but as a "nice to have" — if it fails to install, later code falls back
  to a simpler grader instead of crashing.
- **`detect_platform()`** — looks at the filesystem and environment
  variables to figure out if it's running on Kaggle, molab, Colab, or a
  local machine, and sets the global `PLATFORM` variable to one of those
  four strings. *(This resolves the "where is `PLATFORM` defined?" question
  from earlier — it's a plain global set right here in Cell 1, and every
  later cell that reads it just relies on Cell 1 having already run.)*
- **`detect_devices()`** — asks PyTorch what GPUs exist, how much VRAM
  each has, and whether the GPU supports `bf16` (a lower-precision number
  format that's faster but only works on newer GPUs, Ampere/sm80+ — not
  on Kaggle's older T4s). Stored in the global `DEVICES` dict.
- **`HAS_MATH_VERIFY`** — a global boolean: did the optional math-checker
  library install successfully? Read later by the grading cell to decide
  which math grader to use.

**One-line summary:** *figure out what packages, GPU, and hosting
environment we're on, and store the answers in globals everything else
reads.*

---

## Cell 2 — Registries: the 6 question categories and 5 models

**Big picture:** two lookup tables that describe *what* gets tested and
*on what*. Nothing runs here — it's just data, referenced by config later.

- **`TIER_SPECS`** — one entry per question category ("tier"):
  - **R1** — PopQA, popular/easy trivia questions
  - **R2** — PopQA, obscure/hard trivia questions
  - **R3** — SimpleQA, deliberately tricky trivia
  - **C1** — GSM8K, grade-school math word problems
  - **C2** — MATH, easier competition problems (levels 1–2)
  - **C3** — MATH, harder competition problems (levels 4–5)

  Each entry stores where to download the dataset from, what "family" it
  belongs to (`retrieval` = recall a fact, `reasoning` = work something
  out), what shape the answer takes (`entity`, `numeric`, `latex`...), and
  how many tokens the model is allowed to generate (short for trivia,
  long for math since it needs to show its work).
- **`TIER_FALLBACKS`** — backup dataset sources, tried in order if the
  primary HuggingFace repo is broken, renamed, or gated.
- **`MODEL_SPECS`** — one entry per model in the ladder: four
  instruction-tuned Qwen2.5 sizes (0.5B → 1.5B → 3B → 7B) plus one 7B
  **base** (non-chat) model used specifically to test "does instruction
  tuning change confidence behavior?" (this is H3, in Cell 17). Each entry
  records parameter count, layer count, and hidden-vector width — the
  layer count is what the probe's "5 depth percentiles" (Cell 10) are
  computed against.

**One-line summary:** *the master list of "what questions" (6 tiers) and
"which brains" (5 models) — everything downstream just loops over these
two registries.*

---

## Cell 3 — The `Config` (the master settings panel)

**Big picture:** one Python object (`CFG`) holds *every* setting the whole
pipeline uses. Nothing is hardcoded elsewhere — this is the single source
of truth, and it gets fingerprinted (hashed) so every result can be traced
back to the exact settings that produced it.

### Section-by-section

- **Identity** — run name, random seed, notes. Just labels for tracking.
- **Platform** — `resolved_platform()` picks the real platform name
  (falling back to the Cell-1 auto-detected `PLATFORM` global unless
  overridden) and sets up storage paths accordingly.
- **`STAGES`** — the ordered list of 14 pipeline steps (full breakdown
  below). Removing a name from this tuple skips that stage entirely.
- **Grid subset (`ONLY_*` / `SKIP_*`)** — lets you run only some models or
  only some question categories instead of everything (see dedicated
  section below).
- **Question budgets** — pilot / per-cell / agreement question counts (see
  dedicated section below). Also the train/calibrate/test split ratio
  (60/20/20).
- **Generation settings** — temperature, top-p, batch size, few-shot
  count, how many repeated samples per question (see below).
- **Model execution policy** — how models get loaded (one at a time vs.
  several together) and when to delete downloaded weights to save disk
  (`PURGE_WEIGHTS` — see below).
- **Band gate** — a quality filter: only keep a model/category combo if it
  scores between 25%–80% on the 100-question pilot. Too easy or too hard
  = no real "confidence" signal to study, so it gets dropped.
- **Probe settings (§6)** — controls for the "internal confidence" test.
- **Grading** — how answers get marked right/wrong.
- **Gates** — hard pass/fail thresholds checked at key points before
  trusting results downstream.
- **Statistics** — settings for the final number-crunching.
- **Checkpoint/IO** — auto-resume support, save frequency, what raw data
  to keep.
- **Figures/tables** — output formatting for charts and exported tables.

### `PURGE_WEIGHTS` — cleanup strategy for downloaded model files

Three modes for deleting the big downloaded model files once done with
them (to save disk space):

- **`"never"`** (current setting) — never delete. Simplest, most disk used.
- **`"after_model"`** — delete a model's files right after its *very last*
  use (Cell 22 confirms this fires only on a model's final generation pass,
  never during the pilot pass or when replica-sharding is on, since a
  sibling shard might still be reading the same weights).
- **`"after_run"`** — keep everything until the whole run finishes, then
  wipe the entire weights cache in one go (`purge_all_weights()` in Cell
  11). Middle ground.

*(Separate/unrelated: `EMPTY_CACHE_EVERY_BATCHES` clears GPU memory, not
disk files. The actual GPU-clearing function is `free_cuda()`, defined in
Cell 4.)*

### `SMOKE` mode

A flag that, when on, swaps in a tiny 5-question test version of the whole
config — used to sanity-check every stage runs without crashing before
committing to the real, expensive 1000-question run. Currently `False`.

---

## Model / tier subsetting — `ONLY_*` vs `SKIP_*`

The full grid is **5 models × 6 question categories ("tiers") = 30 cells**.
Four settings narrow that grid down:

- **`ONLY_MODELS`** — a "whitelist." If filled in, *only* those models
  run, everything else is ignored.
  > e.g. `ONLY_MODELS = ("qwen2.5-7b-instruct",)` → run just that one.
- **`SKIP_MODELS`** — a "blacklist." Everything runs *except* what's
  listed.
- **`ONLY_MODELS` wins** if both are set.
- `ONLY_TIERS` / `SKIP_TIERS` work identically for the 6 categories.

**Why this exists:** Kaggle sessions cut off after ~12 hours, so the full
30-cell grid often can't run in one sitting. This lets the work be split
across sessions (small models today, the two 7B ones next session).

---

## Pilot vs. Per-Cell vs. Agreement Check

Key term: a **"cell"** = one specific (model, question-category) pairing.
5 models × 6 categories = **30 cells** total.

| Setting | What it checks |
|---|---|
| `N_PILOT = 100` | Is this model/category combo a reasonable difficulty? |
| `N_AGREEMENT = 100` | Do the 3 confidence-asking styles agree with each other? |
| `N_PER_CELL = 1000` | The real, full-size test once a cell passes the pilot |

- **Pilot** — before the expensive full run on a cell, try 100 questions
  first. Keep only cells scoring 25%–80% accuracy (unless
  `COMMIT_CELLS_OUTSIDE_BAND` forces them in anyway).
- **Agreement check** — the model states confidence 3 different ways
  (percentage / word / bet). Run all three on the same 100-question batch
  and check they agree. Agree → collapse to the best-calibrated one.
  Disagree → the disagreement itself becomes a reportable finding.

## Generation settings block

```python
GREEDY_TEMPERATURE: float = 0.0
SAMPLE_TEMPERATURE: float = 0.8
SAMPLE_TOP_P: float = 0.95
N_SAMPLES: int = 10
N_FEWSHOT_BASE: int = 4
```

- **`GREEDY_TEMPERATURE = 0.0`** — used for normal answer-asking. One
  clean, repeatable answer, not randomness.
- **`SAMPLE_TEMPERATURE = 0.8`** — used *only* for the "ask 10 times"
  step, where randomness is the whole point.
- **`SAMPLE_TOP_P = 0.95`** — a second randomness guardrail.
- **`N_SAMPLES = 10`** — how many times each question is repeated in the
  behavioral-confidence step.
- **`N_FEWSHOT_BASE = 4`** — only for the base 7B model: 4 example Q&A
  pairs shown first, since it isn't instruction-tuned.

## Probe settings (§6) — testing the model's "internal" confidence

- **`PERCENTILES = (0,25,50,75,100)`** — the 5 depth checkpoints inside
  the model where internal snapshots get taken.
- **`PROBE_LABEL = "correct"`** — what the probe predicts: right/wrong.
- **`PROBE_C_GRID`** — tries 4 "strictness levels" while training,
  keeps whichever works best (avoids memorizing noise).
- **`PROBE_STORE_DTYPE = "float32"`** — precision for saved snapshots.
  Kaggle's T4 GPUs can silently glitch numbers in float16 in later
  layers, so float32 avoids that.
- **`AUROC_GATE = 0.65`** — minimum score for the probe to count as
  detecting something real vs. noise ("Gate 3").
- **`LABEL_SHUFFLE_REPEATS = 20`** — builds the random-chance baseline by
  scrambling labels 20 times.
- **`SURFACE_BASELINE = True`** — also checks whether plain TF-IDF text
  matching predicts correctness just as well — if so, the internal probe
  isn't adding anything special.

## Grading settings

- **`NUMERIC_TOLERANCE = 1e-6`** — allow a tiny rounding difference on
  math answers.
- **`USE_NLI_FALLBACK = True`** — backup grading for messy/open-ended
  answers, using `microsoft/deberta-large-mnli` at a 70% entailment
  threshold.
- **`STRIP_ARTICLES = True`** — "the Eiffel Tower" = "Eiffel Tower."

## Gates

- **`GATE1_AGREEMENT = 0.95`** — automated grading must match a human's
  manual check at least 95% of the time.
- **`GATE2_SPEARMAN = 0.60`** — the three confidence-asking styles must
  agree with each other by at least this much.
- **`GATE4_REQUIRE_BASE_ELICITATION = True`** — before comparing base vs.
  instruct models, confirm the base model can produce usable confidence
  answers at all (at least `BASE_ELICITATION_MIN_PARSE_RATE = 0.50`).

## Statistics settings

- **`N_BOOTSTRAP = 2000`**, **`BOOTSTRAP_CI = 0.95`** — resample the data
  2000 times, report a 95%-sure range.
- **`ECE_BINS = 15`**, **`MURPHY_BINS = 10`** — bucket counts for the two
  calibration-honesty measurements.
- **`CALIBRATOR = "auto"`** — isotonic vs. Platt scaling, auto-picked by
  data size (`ISOTONIC_MIN_N = 200`).
- **`MIN_DISTINCT_VERBAL = 3`** — a cell needs at least 3 different
  confidence values from word-based answers, or it's excluded.
- **`QUADRANT_THRESHOLD = 0.5`** — cutoff deciding "high" vs. "low"
  confidence for the hopeful/suppressed quadrant groups.

## Checkpoint / IO settings

- **`RESUME = True`** — an interrupted run picks up where it left off.
- **`CHECKPOINT_EVERY = 50`** — save every 50 questions.
- **`SAVE_RAW_TEXT = True`**, **`SAVE_ACTIVATIONS = True`** — keep full
  answer text and internal snapshots, not just verdicts.

## Figures settings

- **`FIG_DPI = 200`**, **`FIG_FORMATS = ("png", "pdf")`**,
  **`FIG_STYLE = "paper"`** — sharp, dual-format, clean academic look.
- **`LATEX_TABLES = True`** — also export result tables in LaTeX.

---

## The 14 Stages — plain English

Order matters — each stage feeds the next.

1. **`data`** — build the question set (6 categories), split train/cal/test.
2. **`pilot`** — quick 100-question difficulty check per model (25%–80% band).
3. **`verbal`** — ask for confidence 3 ways: percentage, word, bet.
4. **`forced`** — anywhere the model "passed," force it to answer anyway.
5. **`sample`** — ask each question 10 times with randomness on.
6. **`extract`** — grab internal-number snapshots at 5 depths.
7. **`grade`** — mark all answers right or wrong.
8. **`entropy`** — turn the 10 repeated answers into one consistency score.
9. **`probe`** — test whether internal snapshots predict correctness.
10. **`calibrate`** — make all three confidence signals honest probabilities.
11. **`stats`** — run all the statistical comparisons.
12. **`figures`** — generate the charts.
13. **`tables`** — export CSV/LaTeX tables.
14. **`report`** — final write-up: settings, gates, full traceability.

---

## Cell 4 — Paths, Provenance, Checkpointed IO

**Big picture:** the "infrastructure" cell — no AI models run here. It
sets up where files get saved, writes a "receipt" of exactly what
produced this run, locks down randomness for reproducibility, and builds
the save/resume system so a long run survives crashes.

### 1. Folder setup (`PLATFORM_PATHS`, `_resolve_paths`)

A lookup table of "if on Kaggle, save here; if on Colab, save there," etc.
`_resolve_paths()` builds a whole tree of subfolders (raw data,
activations, figures, tables, logs...) inside the run's output folder, so
the rest of the code just references `PATHS["figures"]` instead of typing
paths everywhere. This calls `cfg.resolved_platform()`, which — as Cell 1
confirmed — safely reads the global `PLATFORM` set at notebook start.

### 2. Telling HuggingFace where to save downloads

Sets environment variables so the `transformers` library saves model
downloads into the run's own cache folder instead of the default
location, and applies the API token / offline-mode flag from config.

### 3. The "receipt" for this run (`code_sha`, `build_provenance`)

- **`code_sha()`** — grabs a short fingerprint of the current code version
  from Git. Falls back to `"nogit"` if Git isn't available.
- **`build_provenance()`** — one dictionary: run name, config fingerprint,
  seed, code version, platform/GPU info (`DEVICES`, `HAS_MATH_VERIFY` —
  both set back in Cell 1), start time, exact library versions. Saved to
  disk as `provenance.json` alongside `config.json`, so any result can be
  traced back to exactly what produced it.

### 4. Making randomness repeatable (`set_all_seeds`)

Several libraries (Python's own randomness, NumPy, PyTorch, the GPU) each
keep separate randomness generators. This locks all of them to the same
seed, so re-running the same code twice gives identical results instead
of silently different ones.

### 5. Basic file read/write helpers

- **`jsonl_read` / `jsonl_append`** — JSONL = one JSON record per line.
  `jsonl_read` quietly skips any broken line (in case a session got
  killed mid-write). `jsonl_append` force-writes to disk immediately
  (`fsync`) instead of just buffering in memory — per the code comment:
  *"a checkpoint that is not on disk is not a checkpoint."*
- **`json_write` / `json_read`** — same idea, for one whole JSON file
  (e.g. the provenance receipt).

### 6. The `Checkpoint` class — the actual "resume" mechanism

The real machinery behind `RESUME = True` and `CHECKPOINT_EVERY = 50`.
Works like a to-do-list tracker:

- On startup, reads whatever's already saved and remembers which
  questions are already done (`self.done`).
- `.has(...)` — "have I already processed this one?"
- `.add(...)` — record a new result, buffered in memory, written to disk
  every 50 records instead of one at a time (more efficient).
- `.flush()` — force-write whatever's waiting, even below 50.

**Why it matters:** if a Kaggle session dies at question 743/1000, the
next run resumes at 744 instead of redoing or losing earlier work.

### 7. The `RunLog` class — a running diary

Every important event (a stage starting, a gate passing/failing) gets
written to a log file *and* printed with a timestamp. This becomes the
actual data behind the "run log" report table. Immediately used to log
`"session_start"`.

### 8. Small helper functions

- **`cell_id(model, tier)`** — glues model name + category into one label
  (e.g. `"qwen2.5-7b-instruct__R1"`) — a unique ID for one of the 30 cells.
- **`free_cuda()`** — the actual GPU memory cleanup function.
- **`vram_report()`** — checks current GPU memory used/reserved/peaked —
  useful for debugging out-of-memory issues.

**One-line summary:** *set up organized output folders, write a receipt
of exactly what produced this run, lock down randomness, and build the
save-and-resume system so a long expensive run survives crashes.*

---

# PART 2 — Building the Experiment (Cells 5–8b)

## Cell 5 — Building the question bank (PLAN §3)

**Big picture:** downloads and standardizes questions from 4 real
datasets (PopQA, SimpleQA, GSM8K, MATH) into the 6 tiers, then splits
into train/calibration/test — the actual `data` stage.

- **`hf_load` / `load_with_fallbacks`** — downloads a tier's dataset; if
  the main source fails, tries backups (`TIER_FALLBACKS`) before giving up.
- **Answer-parsing helpers** — each dataset stores its "correct answer"
  differently, so each tier gets its own extractor:
  - `parse_popqa_answers` — pulls *all* acceptable answer variants
    (nicknames/aliases), so grading needs zero fuzzy matching.
  - `gsm8k_answer` — grabs the final number after the `####` marker.
  - `math_answer` — pulls the answer out of a `\boxed{...}` LaTeX command,
    handling nested braces carefully.
  - `math_level` — reads the difficulty level off a MATH problem.
- **`build_tier_rows`** — turns raw data into one standard shape
  (`qid, tier, question, answers, meta`). R1/R2 filter by popularity
  (top/bottom 20%); C2/C3 filter by difficulty level. Rows are always
  shuffled instead of just taking "the first N," to avoid hidden ordering
  bias.
- **`assign_splits`** — shuffles and divides each tier's questions into
  60% train / 20% calibration / 20% test.
- **`build_question_bank`** — the orchestrator:
  - Reuses a previously-built bank from disk if settings match exactly —
    avoids rebuilding every session.
  - Otherwise builds each tier fresh, caps it at 1000 questions, splits
    it, and draws the pilot (100) and agreement (100) subsets **only from
    the train split**, never calibration/test, to avoid data leakage.
  - Saves the finished bank plus a manifest to disk for future reuse.

**One-line summary:** *pull questions from 4 datasets into 6 tiers,
shuffle, split into train/cal/test, carve out pilot + agreement subsets
from train only, cache it all to disk.*

---

## Cell 6 — Building the prompts (PLAN §4, §4.1)

**Big picture:** writes the exact wording sent to the model for every
question style, and forces a strict reply format so answers can be
auto-parsed later with simple pattern matching instead of another AI.

- **`ANSWER_STYLE`** — what shape an answer should take per tier
  (`entity` for PopQA, `numeric` for GSM8K, `latex` for MATH...).
- **`BUCKETS` / `BUCKET_ORDINAL`** — the 5 words for Format B (`CERTAIN`
  → `NO_IDEA`). The ordinal ranking is only for sanity checks — the real
  word→probability mapping gets fit empirically later, not hand-assigned.
- **`BET_GAIN, BET_LOSS = 1, -2`** — Format C's betting payoff: correct
  +1, wrong −2, pass 0. Deliberately unbalanced so passing only becomes
  the smart move once true chance of being right drops below ~67% — turns
  "answer or pass" into a genuine confidence signal.
- **`instruction()`** — builds the instruction text per variant:
  - `A` — answer + confidence as a 0–100 number
  - `B` — answer + confidence as one of the 5 words
  - `C` — answer-or-pass with the betting payoff explained
  - `FORCED` — must answer, no passing/hedging allowed
  - `SAMPLE` / `EXTRACT` — just answer, no confidence talk (confidence is
    measured differently — via repetition or internal snapshots)
- **`FEWSHOT_POOL`** — 4 worked examples per variant, used only for the
  base model.
- **`build_prompt()`** — for instruct models, builds a chat message and
  lets the tokenizer's chat template wrap it; for the base model,
  manually stitches instructions + few-shot examples + the question as
  plain text (base models just continue text patterns, no chat format).

**One-line summary:** *defines the exact question-asking scripts for
every confidence-eliciting style, with a rigid output format for easy
parsing, and prompts instruct vs. base models differently.*

---

## Cell 7 — Parsers: turning free text into structured fields

**Big picture:** the model's raw output is just text — this cell is the
only thing standing between messy sentences and a clean `{answer: ...,
confidence: ...}` record. A parse failure is *recorded as data*, not
thrown away as an error — that's a deliberate design choice, because
"how often does the model fail to follow the format" is itself something
the paper reports on.

- **`key_regex(key)`** — builds (and caches) a regex that finds a line
  like `ANSWER: 42` anywhere in the text, tolerant of bullet points,
  markdown bold, or a leading `>`.
- **`truncate_fewshot_continuation`** — the base model, once shown 4
  examples, doesn't know when to stop — it keeps inventing its own
  follow-up "Question: ... ANSWER: ..." pairs forever. This cuts the text
  at the first self-invented question, so parsing never accidentally
  grabs an answer to a question nobody actually asked. Chat models never
  do this, so it's a safe no-op for them.
- **`grab(text, key)`** — finds every line matching a key and returns the
  *last* match, because reasoning models sometimes restate "ANSWER: X"
  again after working through the problem — the last one is the real one.
- **`clean_answer`** — strips backticks, drops filler like "the answer
  is," keeps only the first line, trims trailing periods.
- **`parse_confidence_numeric`** — pulls a number out of text like "85%"
  or "85" and converts it to a 0–1 fraction; rejects anything over 100.
- **`parse_confidence_bucket`** — matches the raw word to one of the 5
  confidence buckets, tolerating typos and loose phrasing ("Fairly
  confident." → `FAIRLY_CONFIDENT`).
- **`parse_response(variant, text)`** — the main entry point. Never
  raises an exception; instead it always returns a dict with `parse_ok`
  telling you whether the required fields were actually found. Each
  variant (A/B/C/FORCED/SAMPLE) expects different fields, so this
  dispatches to the right combination.

**One-line summary:** *turn free-text model output into clean structured
fields using rigid pattern matching, and record parse success/failure as
a measured quantity instead of silently discarding bad output.*

---

## Cell 8 — Graders: deciding right or wrong

**Big picture:** grading is done in *tiers of increasing effort* — try
the cheap, deterministic method first, and only fall back to a slower AI
model (NLI) when nothing else can decide. Every graded item records
*which* method resolved it, so later you can check "does automated
grading agree with a human?" separately per grading method.

- **`normalize_text`** — lowercases, strips punctuation, optionally drops
  "a/an/the," so "The USA" and "usa" compare equal.
- **`extract_number` / `grade_numeric`** — pulls the final number out of
  an answer and compares it to the gold answer within a small tolerance
  (handles GSM8K).
- **`normalize_math` / `grade_latex`** — strips LaTeX formatting
  (`\left`, `\dfrac`, `\boxed{...}`, etc.) down to bare math. Tries the
  `math_verify` library first (properly checks if two expressions are
  mathematically equal, e.g. `1/2` = `0.5`); if that's unavailable, falls
  back to `sympy` symbolic simplification, and finally to plain string
  comparison.
- **`grade_string`** — exact match after normalization, plus a careful
  "containment" check both ways (only for strings ≥4 characters and
  roughly similar length, so it doesn't wrongly match "Bob" inside
  "Bobby's Diner").
- **`NLIGrader`** — a small entailment model (~1.6GB, loaded only if
  needed) that checks whether the gold answer and the model's answer mean
  the same thing, in *both directions* (bidirectional — "does A imply B"
  AND "does B imply A"), which is stricter than one-way and avoids
  matching a specific answer to a vaguer one that merely contains it.
  Always runs in float32, because this model's attention math genuinely
  breaks under bf16/fp16.
- **`grade_answer()`** — the orchestrator. Tries, in order: exact/alias
  match → numeric match → symbolic math match → NLI fallback (only for
  entity/short answers). Returns `resolved: False` if nothing could
  decide — this is never silently scored as wrong, it's counted
  separately so "unresolvable answers" is itself a visible number.

**One-line summary:** *grade cheaply and deterministically wherever
possible, escalate to a small AI grader only when unavoidable, and always
record which method decided plus whether anything decided at all.*

---

## Cell 8b — Pre-flight compute + storage estimate

**Big picture:** a calculator you run *before* committing to a real
session, so you know roughly how many GPU-hours and how much disk space
the current config will actually cost — before spending 12 hours finding
out the hard way.

- **`HW_PROFILES`** — rough throughput numbers for the three hardware
  setups this project runs on (a big Blackwell GPU, one T4, or two T4s
  split across a model). Bandwidth-bound estimate: roughly, tokens/second
  ≈ GPU memory bandwidth ÷ (2 bytes × number of parameters), scaled by an
  efficiency fudge factor for each setup.
- **`detect_profile()`** — picks the right profile automatically from
  `DEVICES` (Cell 1): big VRAM → Blackwell, multiple small GPUs → T4
  sharded, one small GPU → single T4, no GPU → CPU-ish default.
- **`estimate_compute()`** — for each active model, computes:
  - how many output tokens will actually be generated (3 confidence
    formats + occasional forced-answer + 10 samples + 1 extraction pass,
    all multiplied by tier length and question count)
  - estimated tokens/second on this hardware
  - estimated GPU-hours
  - estimated activation-storage size in MB

  Then prints a table and warns if: some models won't fit on one GPU
  (forcing a slower split across GPUs), or the whole estimate exceeds one
  12-hour session (suggesting `ONLY_MODELS` to split work across
  sessions, since the question bank and checkpoints are shared and
  session 2 can resume cleanly).

**One-line summary:** *before spending real GPU time, estimate how long
the current settings will take and how much disk they'll use, so you can
right-size the run instead of guessing.*

---

# PART 3 — Running the Models (Cells 9–12)

## Cell 9 — Model manager: load, free, and how many at once

**Big picture:** the layer that actually touches the GPU for loading and
unloading models — and the policy for how many models get to share the
GPU at the same time.

- **`auto_batch_size()`** — works out how many questions can be processed
  at once without running out of memory. Estimates: free VRAM after the
  model's own weights, minus a rough per-sequence memory cost that scales
  with layer count, hidden size, and sequence length. The `n_return`
  divisor matters a lot here: asking for 10 samples per question
  (`N_SAMPLES`) effectively multiplies memory cost 10×, so without
  accounting for it the very first batch of the sampling stage would
  always run out of memory.
- **`device_map()`** — one GPU if it fits (molab's single big card);
  `"auto"` (letting `transformers` shard across GPUs) only when there's
  no other choice, like 2×T4 on Kaggle.
- **`LoadedModel`** — a small wrapper bundling the model, its tokenizer,
  its config spec, and which device it actually landed on.
- **`load_model()`** — downloads/loads a model at the right precision
  (bf16 on modern GPUs, fp16 on T4, fp32 on CPU), sets padding correctly,
  and logs how long loading took plus current VRAM usage.
- **`free_model()`** — the explicit teardown: move the model to a
  "meta" device (frees the actual tensors), delete Python references,
  force garbage collection, empty CUDA's memory cache, and — only if
  told to — delete the downloaded weights file from disk to save space.
- **`plan_model_batches()`** — decides which models can share the GPU at
  once. The reasoning: generation is *memory-bandwidth bound* — every
  step has to stream the entire model's weights through the GPU once.
  Two big models running together therefore split the same bandwidth
  instead of adding to it, and also eat into KV-cache memory. Small
  models are the opposite: a 0.5B model alone can't saturate a 96GB card,
  so running several small ones together actually uses the GPU better.
  So: anything above `CONCURRENT_MAX_PARAMS_B` always runs alone; smaller
  models get grouped into "waves."

**One-line summary:** *handles loading a model onto the GPU at the right
precision, cleanly tearing it down afterward (including optionally
deleting its weights), and deciding how many models can share the GPU at
once based on how memory-bandwidth-hungry each one is.*

---

## Cell 10 — Activation extraction: peeking at the model's "thinking"

**Big picture:** this is the machinery behind **internal confidence** —
it captures a snapshot of the model's internal numbers at 5 different
depths, for the very last token of the prompt, right before the model
would start answering.

- **`percentile_layers()`** — converts a percentile (0%, 25%, 50%, 75%,
  100%) into an actual layer index, scaled to however many layers a given
  model has (e.g. a 28-layer model's 50th percentile is layer 14).
- **`ActivationTap`** — registers a **forward hook** at each of those 5
  layers. A forward hook is a small callback PyTorch runs automatically
  every time that layer finishes processing a batch — no need to modify
  the model's own code. Each hook grabs only the *last* token position's
  vector (important: since prompts are left-padded, position `-1` really
  is the true last prompt token for every row in the batch, even though
  they have different lengths), detaches it from the computation graph,
  converts to float32, and moves it to CPU immediately (so it doesn't
  pile up on the GPU).
  - `.pop()` — hands over whatever was captured and clears the buffer for
    the next batch.
  - `.close()` — removes the hooks when done, so they don't keep running
    on future forward passes.
- **`finiteness_stats()`** — a sanity-check function: counts how many of
  the captured numbers are `NaN` or `Inf` (broken numbers, from GPU
  precision issues) versus genuinely finite. This is what later lets the
  probe stage tell "the activations are corrupted" apart from "the
  activations are fine but just don't predict anything" — two very
  different failure modes that would otherwise look identical.

**One-line summary:** *tap the model's internal state at 5 depths for the
last prompt token, using lightweight forward hooks, and separately track
whether those numbers came out clean — this raw capture is what the
probe (Cell 14) later trains on.*

---

## Cell 11 — The generation engine: actually asking the questions

**Big picture:** the workhorse loop. Every stage (pilot, verbal, forced,
sample, extract) funnels through this one engine, which handles batching,
resuming, and automatically recovering from out-of-memory errors.

- **`generate_batch()`** — tokenizes a batch of prompts, runs
  `model.generate()` with the right sampling settings, optionally grabs
  activations via the tap from Cell 10, and decodes only the *newly
  generated* tokens back into text (skipping the prompt itself). If
  `n_return > 1` (the sampling stage), the flat list of outputs gets
  regrouped back into one sub-list per question.
- **`run_generation()`** — the main driver for one (model, tier, variant)
  combination:
  1. Skips anything already checkpointed (`ckpt.has(...)`) — this is the
     resume logic in action.
  2. Picks a batch size via `auto_batch_size()`.
  3. Loops through batches, building prompts and calling
     `generate_batch()`.
  4. **On an out-of-memory error:** clears the CUDA cache, and either
     halves the batch size and retries (if batch size > 1), or — if
     already down to batch size 1 — logs the single question as skipped
     and moves on, rather than getting stuck forever.
  5. For every item, parses the output (Cell 7), builds a record with the
     answer, parse success, config hash, code SHA, and seed for
     traceability, and checkpoints it.
  6. Every so often (`EMPTY_CACHE_EVERY_BATCHES`), clears the GPU cache
     mid-run to prevent memory fragmentation from slowly building up.
  7. When capturing activations, buffers them per-percentile until the
     whole stage finishes, then saves them all at once.
- **`save_activations()`** — writes activation snapshots to a compressed
  `.npz` file per (model, tier). If a file from a previous session
  already exists, it **merges** the two — but crucially, it also
  **de-duplicates by question ID**, keeping only the most recent row per
  question. This matters because the checkpoint's resume logic keys on
  `(qid, variant)`, so if a variant name ever changed, every question
  would look "new" and get appended a second time — silently doubling the
  probe's training data and breaking the statistical independence its
  AUROC score assumes. The de-dup step is the safety net against that.
- **`purge_all_weights()`** — for `PURGE_WEIGHTS = "after_run"`: wipes
  the entire downloaded-weights cache folder in one go at the very end.

**One-line summary:** *the actual "ask the model, handle memory
problems, save the answer, remember what's done" loop that every
generation stage is built on top of — resumable, OOM-adaptive, and
careful never to silently duplicate data.*

---

## Cell 12 — Per-model stage drivers (the generation side)

**Big picture:** five small functions, one per generation stage, that
each decide *which questions* to send into Cell 11's engine and *how*.
This is where the "ragged grid" — not every model/tier combo makes it all
the way through — actually gets implemented.

- **`cell_items(bank, tier, subset)`** — filters a tier's question rows
  down to just the pilot subset, just the agreement subset, or everything.
- **`stage_pilot()`** — 100 questions per active tier, plain FORCED
  answers (no confidence talk), just to measure raw accuracy before
  committing real compute.
- **`stage_verbal()`** — Signal 1. Runs Formats A/B/C. Runs on the
  100-question agreement subset for *every* cell regardless of
  commitment (since the format-agreement test doesn't need a committed
  cell), but only expands to the *full* 1000-question set for cells that
  passed the band gate.
- **`stage_forced()`** — the companion to Format C: wherever the model
  chose to "pass," this forces it to answer anyway (by re-reading its own
  earlier verbal-stage output and pulling out just the pass cases), so
  the analysis can later tell a *justified* hedge from a *missed* answer.
- **`stage_sample()`** — Signal 2. Only for committed cells: asks the
  full question set 10 times each with real randomness on (temperature
  ≥ 0.7 is asserted directly in code — this stage is meaningless without
  actual variance).
- **`stage_extract()`** — Signal 3. Only for committed cells: one greedy
  (temperature 0) pass per question, capturing activations. Deliberately
  reuses the plain SAMPLE-style prompt rather than a confidence-asking
  one, so the probe reads the model's "just answering" internal state,
  not a state shaped by being asked to think about its own confidence.
- **`evaluate_band_gate()`** — grades every model/tier's pilot results
  and decides which cells get committed to the full run: accuracy must
  land in the 25%–80% band (or `COMMIT_CELLS_OUTSIDE_BAND` overrides
  this). Saves the verdict to `cell_commitments.json` — this file is what
  later stages check via `cell_id(...) in committed`.

**One-line summary:** *five thin wrappers that decide exactly which
questions each generation stage should touch, and the gate function that
turns pilot accuracy into a commit/skip decision per cell — this is where
the planned "ragged grid" (not every model×tier survives) actually
happens.*

---

# PART 4 — From Answers to Signals (Cells 13–16)

## Cell 13 — Grading + Semantic Entropy

This cell transforms raw model outputs into ground truth labels and
behavioral confidence. Without it, nothing afterwards can happen.

```
Question → Model Output → Parser → Grading → Correct/Incorrect → Entropy
```

### `stage_grade()`

The grading pipeline. Loops over every stage that produced answers
(pilot, verbal, forced, sample, extract) — because every stage's output
looks a little different (a plain answer, an answer + confidence, 10
sampled answers, an activation-extraction answer) but must all eventually
become one thing: **correct / incorrect**.

- Uses `gold = {qid: question}` (a dictionary) instead of looping through
  the whole question bank for every answer — dictionary lookup is O(1)
  instead of O(N), a big speedup at scale.
- `parsed` can be a single dict (one answer) or a list (10 samples from
  `SAMPLE`) — `plist = ...` normalizes both shapes into one list so the
  same grading loop can handle either.
- Each sample gets graded individually — e.g. Question 52: sample 0
  correct, sample 1 wrong, sample 2 correct... — which later becomes the
  entropy calculation.
- `grade_answer()` (Cell 8) returns a dict with `correct`, `resolved`,
  `grader`, `normalized_answer`. The `**g` syntax expands that whole dict
  directly into the record instead of copying each field by hand.
- Final output: `graded.parquet`. Parquet is used because Pandas, Polars,
  and DuckDB all read it efficiently later.

### `cluster_answers()`

Turns 10 generated answers into semantic clusters — the more interesting
function here.

Naively, 10 different strings look like 10 different answers. But
"United States," "USA," "U.S.," and "America" are the *same* answer. So:

1. **Cheap first pass:** normalize text/math, then group by exact string
   match. String equality is basically free; NLI is O(n²) transformer
   inference and expensive — so cheap clustering happens first, and only
   the *remaining* distinct clusters get compared via NLI.
2. **Bidirectional entailment:** two clusters only merge if each entails
   the other. One-way isn't enough — "dog" entails "animal," but "animal"
   doesn't entail "dog," so they must **not** merge.
3. The merge itself uses **Union-Find** (`parent[]`, `find()`, `union()`)
   — a classic near-O(1) clustering algorithm.

Result: 10 answers become a handful of cluster labels, e.g.
`[0, 0, 0, 1, 1, 2]`.

### `stage_entropy()`

Converts clusters into an uncertainty number.

- All 10 samples in one cluster → entropy 0 → confidence 1 (fully
  consistent).
- 10 samples spread evenly across 10 clusters → entropy is maximal →
  confidence ≈ 0 (pure scatter).
- Formula: `H = entropy(cluster frequencies)`, normalized by
  `Hmax = log(n)` (the theoretical max entropy for `n` equally-likely
  outcomes), so confidence always lands in a clean 0–1 range regardless
  of how many samples were taken.
- `modal_share` — a simpler companion stat: just "what fraction landed in
  the single biggest cluster" (e.g. 7/10 → 0.7). Very interpretable on
  its own.

### `entropy_sanity_check()`

A positive-control test that's easy to forget but important: feed the
entropy calculation an "all identical" case and a "totally uniform" case,
and confirm they come out as confidence≈1 and confidence≈0 respectively.
If not, the entropy math itself has a bug.

---

## Cell 14 — Activation Probes

Asks: **"Does the hidden state already know whether the model will be
correct — before the model has said anything?"** The most novel part of
the pipeline.

- **`load_activations()`** — just loads the saved snapshots
  (`qids`, `layer25`, `layer50`, ...) back from disk.
- **`probe_labels()`** — a subtle but important design decision:
  activations came from the `EXTRACT` stage, so labels should come from
  grading *that same* `EXTRACT` answer — never from a different stage's
  answer to the same question, or you'd be pairing an activation with the
  wrong outcome and injecting label noise. Fallback order (if EXTRACT is
  somehow missing) is EXTRACT → FORCED → SAMPLE, in that order of
  preference.
- **`fit_probe()`** — pipeline: `StandardScaler` → `LogisticRegression`.
  - Scaler, because hidden-layer numbers have wildly different scales,
    and logistic regression becomes unstable without normalizing first.
  - Logistic (not something fancier) *on purpose* — the question isn't
    "can I build the strongest classifier," it's "is the information
    already present in a simple, *linear* form." If plain logistic
    regression works, the information is easily accessible. If only a
    deep MLP could find it, the representation is there but much less
    directly usable.
- **`stage_probe()`** — runs this for every Model × Tier × Percentile
  combination (10 models × 5 layers = 50 probes in the full grid).
  - Splits into train / calibration / test.
  - Uses the **calibration** split (not training, not test) to pick the
    best-performing layer — this prevents any kind of leakage into the
    final reported number.
  - `finite_rows` drops any `NaN` rows first (logistic regression would
    otherwise crash).
  - Uses `predict_proba` (a probability), not the raw correct/incorrect
    prediction, because AUROC needs a ranking, not just a binary call.

### Label shuffle null

One of the strongest controls in the whole notebook. A probe AUROC of
0.73 sounds good — but is it actually meaningful, or could random labels
also score close to that by chance on a small dataset? So: shuffle the
labels, retrain, measure AUROC, repeat 100 times, and look at the 95th
percentile of that random-chance distribution. If the *real* probe beats
that 95th-percentile threshold, that's real evidence — much stronger than
just quoting a bare AUROC number.

### Surface baseline

A second control, asking a different question: "can hidden states
predict correctness, or are some questions just objectively easier no
matter who's answering?" Trains a TF-IDF + Logistic Regression model
using *only the question text*, no model internals at all. If the
surface baseline (e.g. 0.68) is nearly as good as the internal probe
(e.g. 0.69), the hidden states aren't adding much. If surface is 0.56 but
the probe is 0.79, that's strong evidence the internal signal is real.

### `gate3_verdict()`

Final pass/fail decision. Requires **all three**: AUROC above threshold
AND beats the shuffled-label null AND beats the TF-IDF surface baseline.
Intentionally strict. The code also distinguishes "dirty activations"
(NaNs → implementation bug) from "clean activations, no signal"
(→ genuine scientific finding, not a bug) — two very different failure
modes that get logged differently.

---

## Cell 15 — Calibration

Asks: **"When the model says 80%, is it actually right 80% of the
time?"** Accuracy alone doesn't answer that — calibration does.

### `fit_calibrator()`

Two methods, auto-chosen by data size (isotonic needs more data than
Platt):

- **Isotonic Regression** — flexible, no assumed shape (e.g. `0.2 →
  0.15`, `0.5 → 0.42`, `0.8 → 0.91`), needs plenty of data.
- **Platt Scaling** — fits a sigmoid curve with just two parameters,
  works better on small datasets.

### Brier Score

`(predicted_probability − actual_outcome)²`, averaged over examples.
Perfect = 0, worst = 1. Unlike raw accuracy, it directly punishes
overconfidence: being 99% confident and wrong is a huge penalty; being
99% confident and right is nearly free. *(See the note at the very end
of this document — this score is actually symmetric, and underconfidence
gets punished too, just proportionally to how far off it was.)*

### ECE (Expected Calibration Error)

Buckets predictions into bins (0–0.1, 0.1–0.2, ...), then compares
average predicted confidence to average observed accuracy *within each
bin*. Perfect calibration = confidence matches accuracy everywhere.

### Murphy Decomposition

One of the most valuable and overlooked parts of the code. Brier score
alone blends together three different things — Murphy decomposition
separates them:

```
Brier = Reliability − Resolution + Uncertainty
```

- **Reliability** — calibration error itself (lower is better): if the
  model says 0.8, do things actually happen 80% of the time?
- **Resolution** — the model's ability to tell easy questions from hard
  ones by giving them genuinely different probabilities (higher is
  better).
- **Uncertainty** — fixed by the dataset's own base rate (a 50/50 dataset
  is intrinsically harder to score well on than a 95/5 one), nothing to
  do with the model at all.

This lets you answer *why* a Brier score improved — genuinely better
calibration, or just a change in the dataset/accuracy.

### `bootstrap_ci()` / `bootstrap_diff_ci()` / `spearman_with_ci()`

- **`bootstrap_ci()`** — instead of assuming a bell-curve shape,
  resample the dataset with replacement hundreds/thousands of times,
  recompute the statistic each time, and take percentiles of that
  distribution as the confidence range. More robust for things like
  AUROC or Spearman correlation, which don't have simple textbook formulas
  for their uncertainty.
- **`bootstrap_diff_ci()`** — same idea but for the *difference* between
  two systems/signals (`Δ = stat(A) − stat(B)`). If the resulting range
  excludes zero, the difference is unlikely to be random noise.
- **`spearman_with_ci()`** — measures whether, as one confidence signal
  goes up, another tends to go up too (ranking agreement, not exact
  numeric agreement), with a bootstrapped confidence interval around it.

---

## Cell 16 — Signal Assembly

Arguably the most important *engineering* cell in the notebook. Every
earlier cell produced a different kind of confidence measurement, in a
different format. This cell converts all of them into one standardized
table:

```
model | tier | qid | verbal_raw | behavioral_raw | internal_raw |
verbal_cal | behavioral_cal | internal_cal | correct
```

This is the dataframe every hypothesis test afterward uses.

`SIGNALS = ("verbal", "behavioral", "internal")` — one shared list used
everywhere instead of hardcoding the three names repeatedly.

### `empirical_bucket_map()`

Format B's confidence words ("Likely," "Very unlikely"...) aren't
probabilities — they're categories. Rather than hand-assigning "Likely =
0.8," the mapping is learned from data:

- Filtered to `variant == "B"` and `split == "calibration"` only — never
  test, or the mapping would leak information into the eventual score.
- For each (model, tier), and each bucket word: `P(correct | that word)`
  from the actual calibration-split data. E.g. "Likely" said 200 times,
  correct 160 times → `Likely → 0.80`.
- Each model gets its *own* mapping — one model might use "Likely" very
  differently from another.
- **Safeguard:** if a bucket word was never used by a model, its mean
  would be `NaN`. Instead of crashing later, it gets replaced with the
  overall pooled accuracy for that model/tier and flagged `imputed: True`
  ("this bucket never actually occurred").
- Saved as `bucket_mapping.json`.

### `verbal_scores()`

Converts all three verbal formats onto one common 0–1 scale:

- **Format A** — already a number, used as-is.
- **Format B** — look up the empirical bucket map.
- **Format C** — the interesting one. Format C never states a
  probability — just Answer or Pass. So decision theory is used instead:
  ```
  thr = -BET_LOSS / (BET_GAIN - BET_LOSS)
  ```
  With gain +1 and loss −2, that works out to a rational model only
  answering once `P(correct) > 2/3`. An "Answer" only tells us confidence
  is *somewhere above* that threshold — not exactly where — so it's
  approximated with the midpoint between the threshold and 1.0, i.e.
  `ANSWER → (1 + threshold) / 2`. Symmetrically, "Pass" is approximated
  with the midpoint between 0 and the threshold. This is a principled
  approximation, not a pretense of exact knowledge.

### `internal_scores()`

Cell 15's probe only *discovered* the best layer and its AUROC — it
didn't generate a prediction for every single question. This function
does that: picks the best layer by calibration-split AUROC, loads its
activations, rebuilds the same labels, refits the same probe on the
training mask, then calls `predict_proba()` to get an actual `internal_raw`
score (e.g. 0.91, 0.42, 0.12) per question.

### `assemble_signals()`

The function that ties everything together:

1. Convert Format B buckets using the bucket map.
2. Compute verbal scores for all three formats.
3. **Choose the canonical verbal format automatically** — not manually.
   For each format, compute ECE, Brier score, and count of distinct
   values. A model that always outputs "50%" could look perfectly
   calibrated on paper (average confidence = average accuracy) while
   being completely useless (it never distinguishes easy from hard
   questions) — so any format with fewer than `MIN_DISTINCT_VERBAL`
   distinct values gets excluded outright. Among the formats that remain,
   the canonical one is chosen by **smallest Brier score**, not smallest
   ECE (the code comments explain the reasoning is that Brier is harder
   to game by being uselessly constant).
4. Merge in behavioral confidence, internal confidence, and ground truth
   — now one row per (model, tier, question) holds all three signals.
5. **Calibrate every signal separately**, per (model, tier), fitting a
   fresh isotonic/Platt calibrator each time — using *only* the
   calibration split, never test.
6. **Degeneracy removal** — if a verbal signal ends up with only 2
   distinct values after everything, it's wiped to `NaN` entirely, so the
   pipeline never reports "perfect calibration" for a predictor that's
   secretly just a constant.
7. Everything gets written to `signals.parquet` plus
   `calibration_meta.json`.

**One-line summary:** *the bridge between raw experimental outputs and
statistical testing — standardizes three fundamentally different
confidence signals onto one comparable 0–1 scale, calibrates each fairly
using only the calibration split, removes any predictor that turned out
degenerate, and produces the one table every later hypothesis test relies
on.*

---

# PART 5 — Testing the Hypotheses (Cells 17–18)

## Cell 17 — Hypothesis Tests

If Cell 16 prepared the evidence, this is where the paper actually tests
its scientific claims. Every function corresponds to a pre-registered
hypothesis (H0–H4) or a supporting analysis.

### `test_h0_format_agreement()`

**Question:** are the three verbal formats (A/B/C) actually measuring
the same underlying thing? Filters to the agreement subset, reshapes so
each question has columns A/B/C, and computes pairwise Spearman
correlations (ranking-based, since one format might consistently run
higher than another but still preserve the same *order*).

The important part: `gate2_pass = min(lower_bounds) >= threshold` — not
just the average correlation, but the *lower bound of the bootstrapped
confidence interval* for the weakest pair must clear the threshold. A
much stronger bar than reporting a single average number. If it fails,
later analysis reports all three verbal formats separately rather than
collapsing them into one.

### `test_h1_signal_calibration()`

**Question:** which of the three signals (verbal / behavioral /
internal) is calibrated best? Test-split only. Computes ECE, Brier, and
Murphy decomposition for each signal, ranks them by ECE, then
bootstraps the difference between the best and worst.

Key subtlety: the hypothesis specifically predicts *verbal confidence is
the worst-calibrated one*. If behavioral turns out worse than verbal
instead, the signals genuinely differ — but the *specific* hypothesis is
still declared falsified, because it requires **both** a real difference
AND the predicted direction to match.

### `question_features()`

Extracts simple, deliberately-interpretable properties of each question
(word count, contains a year, contains digits, multiple capitalized
entities, question family). Kept intentionally simple — these were
pre-registered as "cheap" predictors for H2, not sophisticated language
features.

### `test_h2_quadrants()`

Builds the "confidence quadrants" this project is centered on. Behavioral
and internal confidence are averaged into `other_cal`; every question
then falls, by threshold, into one of four regions:

| Verbal | Other signals | Label |
|---|---|---|
| High | Low | **hopeful** (performed confidence) |
| Low | High | **suppressed** (real knowledge, hedged in words) |
| High | High | agree_high (genuine confidence) |
| Low | Low | agree_low (genuine uncertainty) |

To test whether these quadrants are *meaningfully* linked to question
properties (not random), a chi-square test is run — but the code doesn't
stop there. It repeatedly shuffles the feature labels and recomputes
chi-square to build a null distribution, and requires the observed
statistic to beat **both** classical significance **and** the
shuffle-based null — a stronger bar than an ordinary chi-square test
alone. Representative hopeful/suppressed examples get saved for
qualitative write-up.

### `abstention_split()`

A "pass" under Format C can mean two very different things, only
distinguishable once the forced-answer follow-up comes back:

- Forced answer turns out **correct** → the model actually knew, but
  hedged anyway → **missed knowledge**.
- Forced answer turns out **wrong** → the hedge was appropriate →
  **justified hedge**.

Without this split, simply counting "how often did it pass" would
confuse a cautious-but-competent model with a genuinely ignorant one.

### `omniscience_index()`

One summary score per model:

```
(correct − incorrect) / total × 100
```

Abstentions count as zero — they neither help nor hurt. Correct answers
push the score up, wrong answers push it down. Summarizes how effectively
a model answers while avoiding outright errors.

### `test_h3_base_vs_instruct()`

Compares the base 7B model to the instruction-tuned 7B model. Hypothesis:
instruction tuning should reduce "hopeful" confidence. Measures the
hopeful rate for both, bootstraps the difference — but adds a safeguard:
if instruction tuning simply refuses to answer *more often* (rather than
becoming better calibrated), that's not really an improvement, so
"missed knowledge rate" is checked too, guarding against mistaking
excessive caution for genuinely better calibration.

### `test_h4_depth()`

Uses the layer-by-layer AUROC sweep from Cell 14/15: for each model,
finds the earliest layer percentile where AUROC first crosses the
threshold (the "onset" depth). The prediction is that models built for
multi-step reasoning show a *later* onset than models built for simple
retrieval. Bootstrapped confidence intervals compare the onset
distributions between model types, and onset-vs-parameter-count is
optionally examined too.

### `hierarchical_regression()`

Instead of fitting many small separate regressions, one global
statistical model is fit: outcome = correct; predictors = verbal
confidence, behavioral confidence, internal confidence, reasoning family,
parameter count, layer depth, and their interactions. Ideally a Bayesian
mixed-effects logistic regression with question-level random effects; if
that fails to fit, it automatically falls back to clustered logistic
regression. Gives interpretable coefficients for how each signal
predicts correctness while accounting for model family and question
structure.

### `correlation_table()`

Reports relationships between the three signals — deliberately using
**Spearman** for raw (uncalibrated) scores, since raw scores may not be
linearly related, and **Pearson** for calibrated scores, since after
calibration the numbers are meaningful probabilities where linear
correlation is the right tool. Saved per-model and pooled across all
models.

---

## Cell 18 — Post-hoc LLM Judge

Not part of the main experiment — an **audit layer**. The deterministic
grader (Cell 8) remains the official source of truth throughout. This
cell just asks: if a strong LLM independently re-judges the same
answers, how often does it agree? That agreement rate becomes **Gate 1**
of the experimental validation.

- **`JudgeConfig`** — whether the audit is enabled, which model to use
  (with fallbacks), batching, and *which* examples get audited — not
  everything, to keep it computationally manageable. Targets can include
  unresolved deterministic cases, fuzzy semantic matches, a random audit
  sample, or the whole dataset.
- **`JUDGE_PROMPT`** — deliberately minimal: compare prediction to
  reference, ignore wording differences, judge only semantic
  equivalence, reply with exactly `VERDICT: CORRECT` or `VERDICT:
  INCORRECT`. The rigid format makes parsing reliable.
- **`load_judge()`** — auto-detects what the environment supports
  (native BF16, AWQ, GPTQ, 4-bit) and, if one checkpoint fails to load,
  automatically tries progressively smaller fallback models — keeping
  the audit portable across different GPUs.
- **`judge_targets()`** — selects which examples to audit (unresolved
  answers, NLI/string-based grades, or a balanced random sample per
  answer family), de-duplicating before inference.
- **`stage_judge()`** — the main audit pipeline: checks a checkpoint for
  what's already judged (resumable), builds prompts, runs generation in
  batches, parses each output into `True / False / Unknown`, and keeps
  the raw judge text too for later inspection. Once finished, computes
  overall agreement plus agreement broken down by answer form and by
  which deterministic grading strategy originally decided it. Finally
  checks whether the agreement rate clears the pre-registered Gate 1
  threshold.

  Important: even perfect judge agreement does **not** replace human
  validation — the notebook explicitly still requires a manual sample
  check.
- **`export_manual_check_sheet()`** — builds the human-audit spreadsheet:
  a fixed number of random examples per answer family, each with
  question, gold answer, model answer, automated label, and grading
  strategy, plus empty columns for a human annotator to fill in manual
  correctness and disagreement notes. This CSV is the final step needed
  to actually close Gate 1.

---

# PART 6 — From Results to a Paper (Cells 19–23)

Everything from here on happens *after* the models have finished
generating answers. This is the part that turns raw experiment output
into a publishable paper.

```
Model Outputs → Grade Answers → Compute Signals → Statistical Tests →
Figures → Tables → Final Report → Paper-ready artifacts
```

## Cell 19 — Figure Generation

Turns statistical results into publication-quality figures. Nothing is
trained or graded here — everything has already happened; this cell only
visualizes it.

- **`PALETTE`, `TIER_COLOR`** — a shared color theme, with each tier
  fixed to *its own* color always (e.g. Tier 2 is always orange, even in
  a figure that doesn't show Tier 1) — otherwise many plotting libraries
  would silently reassign colors depending on which subset is plotted,
  which is confusing across figures.
- **`apply_style()`** — sets matplotlib defaults globally (white
  background, light grids, small fonts, "paper" style, publication DPI)
  so every figure looks consistent.
- **`save_fig()`** — saves each figure in every configured format
  (png/pdf/svg) plus a caption text file, and logs that it happened.
- **`reliability_curve()`** — bins predictions (e.g. 0.1–0.2, 0.2–0.3...)
  and computes average predicted confidence vs. average true accuracy per
  bin — the calibration curve itself.

**Figure 1** (H1) — predicted confidence vs. actual accuracy for all
three signals, with a shaded 300-sample bootstrap confidence band. A
second panel breaks this into the Murphy decomposition (reliability vs.
resolution) to explain *why* calibration differs, not just that it does.

**Figure 2** (H2) — the confidence quadrants: x = verbal confidence, y =
behavioral/internal confidence, one point per question, four visible
regions. A second subplot shows justified-hedge / missed-knowledge /
unresolved from the abstention analysis.

**Figure 3** (H3) — base vs. instruction-tuned model, comparing hopeful
confidence and missed knowledge, with confidence-interval bars.

**Figure 4** (H4, probably the most interesting) — probe AUROC plotted
against layer depth (5% → 95%), one curve per reasoning tier, with a gray
shuffled-label "chance" band — anything inside that band isn't
meaningful. If reasoning genuinely only appears late in the network, the
curve stays flat and then rises near the end, exactly as H4 predicts.

**Figure 5** — pilot accuracy for every (model × tier) cell as a grid of
squares; any square outside the 25–80% band gets a dotted border,
visualizing the commitment gate directly.

**Figure 6** — a correlation heatmap: rows = model+tier, columns =
verbal-vs-behavioral / verbal-vs-internal / behavioral-vs-internal,
colored blue (negative) → white (zero) → red (positive).

- **`stage_figures()`** — the orchestrator: loops through every figure
  function, wrapping each in try/except so one broken figure never stops
  the rest from being generated.

---

## Cell 20 — Table Export

Figures are for humans; tables are for the paper itself.

- **`export_table()`** — given a dataframe, writes CSV, Parquet, and
  (if enabled) a ready-to-paste LaTeX table via `df.to_latex()` — one
  dataframe, three output formats.
- **`stage_tables()`** — generates all named tables, including:
  - **T1** — dataset composition (tier, difficulty, source, train/cal/test)
  - **T2** — cell commitments (which model×tier combos were kept)
  - **T3** — parsing success and accuracy/resolution
  - **T5** — H0 results
  - **T6** — Murphy decomposition
  - **T7** — probe AUROC
  - **T8** — depth onset (H4)
  - **T9** — correlation table
  - **T10** — Omniscience Index
  - **T11** — abstention statistics
  - **T12** — per-question master table (every signal, every prediction,
    every question) — essentially the whole appendix dataset in one file

---

## Cell 21 — Final Report

- **`compute_ledger()`** — reads `events.jsonl`, finds `stage_timing`
  events, and computes actual measured GPU-hours from them instead of
  estimating.
- **`stage_report()`** — one large dictionary covering the grid,
  question counts, hypothesis verdicts, compute usage, artifacts
  produced, judge audit results, entropy sanity check, regression
  results, and gate status. Saved as `final_report.json`, plus
  `run_log_rows.md` — a table literally ready to paste into the PLAN
  document.

---

## Cell 22 — Main Driver

The heart of the entire pipeline — everything funnels through
`run_pipeline()`, in phases:

1. **Build the question bank** — only once, reused across models.
2. **Load the NLI grader** — used later for grading.
3. **Pilot pass** — every active model answers enough questions to
   estimate accuracy per cell.
4. **Band gate** — only cells with acceptable pilot accuracy get
   committed; this is what prevents wasting compute on cells that are
   too easy or too hard to say anything meaningful about confidence.
5. **Heavy generation** — every committed model generates Verbal,
   Forced, Sampled, and Extraction outputs, model by model, in "waves"
   decided by `plan_model_batches()` (Cell 9).
6. **GPU cleanup** — unload the model, free VRAM, repeat for the next
   wave — this is how several large models run sequentially on limited
   hardware without ever holding more than they need in memory at once.
   Weight-purging (if `PURGE_WEIGHTS = "after_model"`) is only triggered
   here, and only on a model's genuinely final pass — the pilot pass
   never purges, because the same model gets loaded a second time for the
   heavy-generation stage, and purging after the pilot would just force a
   wasteful re-download.
7. **CPU analysis** — now that all generation is done, compute Grading,
   Entropy, Probes, Signals, Calibration, and Statistics — none of this
   needs the GPU (aside from the small NLI grader).
8. **Hypothesis tests** — H0 through H4 all run here.
9. **Judge audit** — optional, loads last (after everything GPU-heavy is
   already freed), and frees itself when done.
10. **Figures** → **11. Tables** → **12. Report** — the publication
    pipeline described above.

Finally: free the GPU one last time, purge weights entirely if
`PURGE_WEIGHTS = "after_run"`, and return every intermediate result
(bank, commitments, graded data, signals, all hypothesis results,
correlation table, Omniscience Index, abstention stats, Gate 3 verdict,
judge results, and the final report) in one dictionary.

---

## Cell 23 — Run

```python
RESULTS = run_pipeline(CFG, JUDGE)
```

Executes the entire experiment. Everything built across the previous 22
cells ultimately funnels through this single call. When it finishes, it
prints a concise summary: how many experimental cells were committed
after the pilot gate, total measured GPU-hours, the status of every
validation gate, the verdicts for H0–H4, where figures/tables were
written, and the next manual step (filling in the Gate 1 audit sheet).

### The overall architecture, one more time

```
Raw model outputs
        │
        ▼
Grade answers
        │
        ▼
Compute confidence signals (verbal / behavioral / internal)
        │
        ▼
Run statistical analyses (H0–H4)
        │
        ▼
Generate publication-quality figures
        │
        ▼
Export publication-ready tables
        │
        ▼
Measure compute and summarize the run
        │
        ▼
Write a final report + reproducibility artifacts
        │
        ▼
Print a concise experiment summary
```

---

## Previously-open questions — now resolved

- **Where is `PLATFORM` defined?** Cell 1: `PLATFORM = detect_platform()`,
  a plain module-level global. `resolved_platform()` (Cell 3) just reads
  it. Not actually a bug — it only works because Cell 1 runs before
  everything else, which is guaranteed in a normal top-to-bottom run.
- **Where does `purge_weights` actually get called, and how does it know
  a "final pass" is really final?** Cell 22: `purge=True` is only passed
  during the *heavy-generation* pass (never the pilot pass, since every
  model is deliberately loaded twice), and only when there's no replica
  sharding in play — `len(jobs) == len(wave)`. With sharding, the last
  shard to finish could otherwise delete a weights snapshot a sibling
  shard is still reading from disk.
- **What does `plan_model_batches` do?** Cell 9 — groups models into
  "waves" that can share a GPU, based on parameter size and a configured
  concurrency limit (see Cell 9 section above).
- **Where do `DEVICES` and `HAS_MATH_VERIFY` come from?** Both set in
  Cell 1, as globals, before `Config` or anything else is defined.
- **Where are `TIER_SPECS` and `TIER_FALLBACKS` defined?** Cell 2,
  alongside `MODEL_SPECS`.

---

## A note on the Brier score question

*(worth keeping here since it came up while reading Cell 15/19)*

**"If Brier score penalizes overconfidence, shouldn't it also penalize
underconfidence — especially when the answer is right?"**

It already does — the formula is symmetric, not one-sided:

```
Brier = (predicted_probability − actual_outcome)²
```

`actual_outcome` is 1 if correct, 0 if wrong. The penalty only depends on
*how far* the prediction sits from that 0 or 1 — not on which direction
it's off in. A few worked examples make this concrete:

| Predicted p | Actual | Penalty | Read |
|---|---|---|---|
| 0.99 | wrong (0) | 0.980 | overconfident + wrong → huge penalty |
| 0.50 | right (1) | 0.250 | underconfident + right → real penalty |
| 0.50 | wrong (0) | 0.250 | underconfident-in-the-other-direction + wrong → **identical** penalty |
| 0.99 | right (1) | 0.0001 | confident + right → almost free |

Rows 2 and 3 are the key comparison: saying "50%" costs exactly the same
0.25 penalty whether the answer turns out right or wrong, because the
*distance* from 0.5 to either 0 or 1 is identical. So underconfidence on
a correct answer is absolutely being punished — it's just numerically
smaller than the penalty for confident-and-wrong, because 0.5 is *closer*
to 1 than 0.99 is to 0. That's not an asymmetry in the scoring rule
itself — it's just what "distance" looks like at different points on the
scale. ECE and the Murphy "reliability" term work the same way: both are
based on `|confidence − accuracy|`, which treats over- and under-shooting
identically.

Where this actually matters for this project: the **suppressed
confidence** quadrant in H2 (Cell 17) is specifically the underconfidence
case — real internal/behavioral knowledge, hedged in words. Brier/ECE on
the verbal signal alone would already reflect that as a calibration
penalty; the quadrant analysis exists on top of that mainly to say *why*
it happened (verbal vs. internal disagreement) rather than to catch
underconfidence that the symmetric loss would otherwise miss — it
already wouldn't.

If, for a specific research argument, over- and under-confidence should
be weighted *differently* on purpose (e.g. "in this domain, an
underconfident hedge is cheaper than an overconfident wrong answer"),
that requires a genuinely asymmetric loss function — Brier itself won't
do it, since squared error is symmetric by construction. That would be a
deliberate deviation from the standard metric, worth flagging explicitly
in the paper rather than assuming Brier already encodes it.
