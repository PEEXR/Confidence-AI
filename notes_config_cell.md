# Decoding My Research Pipeline — Notes

> Personal reference notes while reading through the LLM confidence
> research codebase (Qwen2.5 confidence signals project). Updated as I
> go through each cell.

---

## Glossary (terms I keep forgetting)

| Term | Plain-English meaning |
|---|---|
| **Weights** | The actual downloaded model file (the "brain"). Sizes range ~1GB–14GB depending on model size. |
| **Dataclass** | A Python shortcut for a class that's basically just a bundle of settings/fields. |
| **VRAM** | GPU memory (separate from normal disk/RAM). |
| **Temperature** | Controls randomness when the AI generates text. `0` = always the same, most likely answer. Higher (e.g. `0.8`) = more varied/random answers. |
| **Greedy decoding** | Temperature = 0. Model always picks its single most likely next word — deterministic. |
| **Sampling** | Temperature > 0. Model can pick less-likely words sometimes — gives variety across repeated runs. |
| **Top-p** | A second randomness guardrail — only considers the top X% most-likely words when picking randomly, so sampling doesn't produce total nonsense. |
| **Few-shot** | Showing the model a few example Q&A pairs before the real question, so it understands the expected format. Mainly needed for base models that aren't instruction-tuned. |
| **Hidden state / activation** | A snapshot of the model's internal numbers while it's processing — like peeking at its "thinking" mid-process, before it even finishes an answer. |
| **Probe** | A small, simple classifier trained on those internal snapshots to see if they secretly predict something (like "is the model about to be right or wrong?"). |
| **AUROC** | A score (0.5 = random guessing, 1.0 = perfect) measuring how well something predicts an outcome. Used here to check if the probe is actually picking up a real signal. |
| **Label shuffle / null baseline** | Scrambling the right/wrong labels on purpose to see how well a probe could "predict" by pure chance — the real probe's score gets compared against this. |
| **TF-IDF / surface baseline** | A simple, old-school text-pattern-matching method (no deep "understanding"), used as a control — if this dumb method predicts correctness just as well as the fancy internal probe, the probe isn't adding much. |
| **NLI (Natural Language Inference)** | An AI model trained to judge whether two sentences mean the same thing, even worded differently. Used here as a backup grading method. |
| **Calibration** | Making a confidence score "honest" — e.g. making sure things the model calls "80% confident" really are right ~80% of the time. |
| **ECE (Expected Calibration Error) / Brier score** | Numbers that measure how "honest" a confidence score is overall. |
| **Murphy decomposition** | A way of breaking a calibration score into separate parts (how honest / how discriminating / how hard the task is) instead of one blended number. |
| **Spearman correlation** | A number showing how much two things move together/agree in ranking. |
| **Bootstrap / confidence interval (CI)** | A statistics trick: resample your data many times to estimate how much uncertainty is in a result, then report a range you're fairly sure the true value falls in. |
| **Checkpointing** | Regularly saving progress so a long job can resume instead of restarting from zero if interrupted. |
| **Provenance** | A "receipt" recording exactly what settings/code/versions/time produced a given result, so it's traceable later. |
| **Seed** | A starting number for randomness generators. Locking it means "random" results are reproducible — same seed, same outcome every time. |
| **JSONL** | A text file format where each line is its own separate JSON record — easy to append to one line at a time, good for long/resumable jobs. |
| **Git SHA** | A short ID fingerprinting the exact version of the code, so you know exactly what code produced a given result. |

---

## Cell 3 — The `Config` (the master settings panel)

**Big picture:** One Python object (`CFG`) holds *every* setting the whole
pipeline uses. Nothing is hardcoded elsewhere — this is the single source
of truth, and it gets fingerprinted (hashed) so every result can be traced
back to the exact settings that produced it.

### Section-by-section

- **Identity** — run name, random seed, notes. Just labels for tracking.
- **Platform** — auto-detects where it's running (Kaggle / Colab / local)
  and sets up storage paths accordingly.
  - ⚠️ Found a possible bug here: `resolved_platform()` references a bare
    `PLATFORM` variable that isn't defined in this cell — and Cell 4 relies
    on this method too (`_resolve_paths` calls it directly). Still need to
    check earlier cells for where `PLATFORM` might be set, or whether this
    throws a `NameError` on the default `"auto"` setting.
- **`STAGES`** — the ordered list of 14 pipeline steps (full breakdown below).
  Removing a name from this tuple skips that stage.
- **Grid subset (`ONLY_*` / `SKIP_*`)** — lets you run only some models or
  only some question categories, instead of everything (see dedicated
  section below).
- **Question budgets** — pilot / per-cell / agreement question counts (see
  dedicated section below). Also the train/calibrate/test split ratio
  (60/20/20).
- **Generation settings** — temperature, top-p, batch size, few-shot count,
  how many repeated samples to take per question (see dedicated section
  below).
- **Model execution policy** — how models get loaded (one at a time vs.
  multiple at once) and when to delete downloaded weights to save disk
  space (`PURGE_WEIGHTS` — see below).
- **Band gate** — a quality filter: only keep a model/category combo if it
  scores between 25%–80% on the 100-question pilot. Too easy or too hard
  = not useful for measuring confidence, so it gets dropped.
- **Probe settings (§6)** — controls for the "internal confidence" test.
- **Grading** — how answers get marked right/wrong.
- **Gates** — hard pass/fail thresholds the pipeline checks at key points
  before trusting results downstream.
- **Statistics** — settings for the final number-crunching.
- **Checkpoint/IO** — auto-resume support, how often to save progress,
  what raw data to keep.
- **Figures/tables** — output formatting for charts and exported tables.

*(Probe / grading / gates / statistics / checkpoint / figures sections are
broken down field-by-field further down this doc.)*

### `PURGE_WEIGHTS` — cleanup strategy for downloaded model files

Three modes for deleting the big downloaded model files once you're done
with them (to save disk space):

- **`"never"`** (current setting) — never delete anything. Simplest, but
  uses the most disk space since all 5 models' files sit around the whole
  time.
- **`"after_model"`** — delete a model's files right after its *very last*
  use in the whole run (riskier — could accidentally delete something
  still needed, forcing a re-download).
- **`"after_run"`** — keep everything until the *entire* run finishes,
  then delete all of it in one go at the end. Middle ground: simpler and
  safer than `after_model`, but still uses more disk than deleting as you go.

*(Separate/unrelated: `EMPTY_CACHE_EVERY_BATCHES` clears GPU memory, not
disk files — different resource entirely, easy to confuse with the above.
The actual GPU-clearing function is `free_cuda()`, defined in Cell 4.)*

### `SMOKE` mode

A flag (`SMOKE = True/False`) that, when on, swaps in a tiny 5-question
test version of the whole config — used to sanity-check that every stage
runs without crashing, before committing to the real, expensive
1000-question run. Currently `False`, so the real config is active.

---

## Model / tier subsetting — `ONLY_*` vs `SKIP_*`

The full grid is **5 models × 6 question categories ("tiers") = 30 cells**.
These four settings narrow that grid down:

- **`ONLY_MODELS`** — a "whitelist." If filled in, *only* those models run,
  everything else is ignored completely.
  > e.g. `ONLY_MODELS = ("qwen2.5-7b-instruct",)` → run just that one model.
- **`SKIP_MODELS`** — a "blacklist." Everything runs *except* what's listed.
  > e.g. `SKIP_MODELS = ("qwen2.5-7b-instruct", "qwen2.5-7b-base")` → run
  > the other 3, skip the two 7B ones.
- **`ONLY_MODELS` wins** if both are set — `SKIP_MODELS` becomes irrelevant
  once you've already narrowed to a specific list.
- `ONLY_TIERS` / `SKIP_TIERS` work identically, but for the 6 question
  categories instead of models.

**Why this exists:** Kaggle sessions cut off after ~12 hours, so the full
30-cell grid often can't run in one sitting. This lets you split the work
across sessions (e.g. small models today, the two 7B ones next session) —
exactly what the commented-out presets at the bottom of Cell 3 do.

---

## Pilot vs. Per-Cell vs. Agreement Check

Key term: a **"cell"** = one specific (model, question-category) pairing.
5 models × 6 categories = **30 cells** total.

- **Pilot (`N_PILOT = 100`)** — Before running the expensive full test on a
  cell, try it on 100 questions first. Keep only cells scoring 25%–80%
  accuracy — too easy or too hard means no real "confidence" signal to
  study. Cells outside the band get dropped (unless
  `COMMIT_CELLS_OUTSIDE_BAND` forces them in anyway).
- **Per-cell (`N_PER_CELL = 1000`)** — Once a cell passes the pilot, this is
  how many questions it gets in the real, full run.
- **Agreement check (`N_AGREEMENT = 100`)** — The model states confidence 3
  different ways (percentage / word / bet). Before trusting any one of
  these, run all three on the same 100-question batch and check: do they
  actually agree with each other? If yes → collapse to the
  best-calibrated one. If no → the disagreement itself becomes a reportable
  finding.

| Setting | What it checks |
|---|---|
| `N_PILOT = 100` | Is this model/category combo a reasonable difficulty? |
| `N_AGREEMENT = 100` | Do the 3 confidence-asking styles agree with each other? |
| `N_PER_CELL = 1000` | The real, full-size test once a cell passes the pilot |

---

## Generation settings block

```python
GREEDY_TEMPERATURE: float = 0.0
SAMPLE_TEMPERATURE: float = 0.8
SAMPLE_TOP_P: float = 0.95
N_SAMPLES: int = 10
N_FEWSHOT_BASE: int = 4
STOP_ON_DOUBLE_NEWLINE: bool = False
```

- **`GREEDY_TEMPERATURE = 0.0`** — used for normal answer-asking (confidence
  formats + the internal-snapshot step). Temperature 0 on purpose: one
  clean, consistent, repeatable answer, not randomness.
- **`SAMPLE_TEMPERATURE = 0.8`** — used *only* for the "ask 10 times" step.
  Randomness is wanted here — the whole point is seeing whether answers
  stay consistent or scatter. Plan requires 0.7–1.0; 0.8 fits.
- **`SAMPLE_TOP_P = 0.95`** — a second randomness guardrail, works together
  with temperature so random picks don't get too weird.
- **`N_SAMPLES = 10`** — how many times each question gets repeated in the
  "ask again and again" step. Matches the plan's requirement exactly.
- **`N_FEWSHOT_BASE = 4`** — only for the base (non-instruct) 7B model:
  shows it 4 example Q&A pairs first, like a mini cheat-sheet, so it
  understands the expected answer format. Instruct models don't need this.
- **`STOP_ON_DOUBLE_NEWLINE = False`** — a rule for cutting off generation
  early if the model produces two blank lines in a row. Currently off.

---

## Probe settings (§6) — testing the model's "internal" confidence

- **`PERCENTILES = (0,25,50,75,100)`** — the 5 depth checkpoints inside the
  model where internal snapshots get taken.
- **`PROBE_LABEL = "correct"`** — what the probe tries to predict: plain
  right/wrong (`"correct"`) or the 10-repeat consistency score
  (`"entropy"`). Currently set to right/wrong.
- **`PROBE_C_GRID = (0.01, 0.1, 1.0, 10.0)`** — tries 4 different
  "strictness levels" while training the probe, keeps whichever works best
  (avoids the probe just memorizing noise).
- **`PROBE_MAX_ITER = 2000`** — cutoff: probe gets at most 2000 training
  rounds before the code gives up and moves on.
- **`PROBE_STORE_DTYPE = "float32"`** — precision used to save internal
  snapshots. Ties to a known risk: Kaggle's T4 GPUs can silently glitch
  numbers in a lower-precision format (float16) in later layers — float32
  avoids that.
- **`AUROC_GATE = 0.65`** — minimum score for the probe to count as
  "detecting something real" vs. noise. This is "Gate 3."
- **`LABEL_SHUFFLE_REPEATS = 20`** — builds the "random chance" baseline by
  scrambling labels 20 times; the real probe gets compared against this.
- **`SURFACE_BASELINE = True`** — also checks whether a much simpler
  text-pattern method (TF-IDF) predicts correctness just as well — if so,
  the internal probe isn't adding anything special.

---

## Grading settings

- **`NUMERIC_TOLERANCE = 1e-6`** — allow a tiny rounding difference on math
  answers instead of demanding an exact digit match.
- **`USE_NLI_FALLBACK = True`** — turns on the backup grading method for
  messy/open-ended answers.
- **`NLI_MODEL = "microsoft/deberta-large-mnli"`** — the specific AI model
  used for that backup check (judges if two answers mean the same thing).
- **`NLI_ENTAIL_THRESHOLD = 0.70`** — how confident that backup model needs
  to be (70%) before counting two answers as a match.
- **`NLI_BATCH_SIZE = 64`** — efficiency setting: how many answers get
  checked at once by the backup model.
- **`STRIP_ARTICLES = True`** — removes filler words like "a/an/the" before
  comparing answers, so "the Eiffel Tower" = "Eiffel Tower."

---

## Gates

- **`GATE1_AGREEMENT = 0.95`** — automated grading must match a human's
  manual check at least 95% of the time.
- **`GATE2_SPEARMAN = 0.60`** — the three confidence-asking styles must
  agree with each other by at least this much.
- **`GATE4_REQUIRE_BASE_ELICITATION = True`** — before comparing base vs.
  instruct models, first confirm the base model can produce usable
  confidence answers at all.
- **`BASE_ELICITATION_MIN_PARSE_RATE = 0.50`** — the actual bar: at least
  50% of the base model's confidence answers must come back understandable.

---

## Statistics settings

- **`N_BOOTSTRAP = 2000`** — resample the data 2000 times to estimate
  uncertainty in a result (the "bootstrap" trick).
- **`BOOTSTRAP_CI = 0.95`** — report a range you're 95% sure contains the
  true value.
- **`ECE_BINS = 15`** — number of buckets used when measuring how "honest"
  confidence scores are (ECE).
- **`MURPHY_BINS = 10`** — same idea, but for the separate Murphy
  decomposition analysis — 10 buckets.
- **`CALIBRATOR = "auto"`** — which method makes confidence scores honest
  (isotonic vs. Platt); "auto" picks based on how much data is available.
- **`ISOTONIC_MIN_N = 200`** — the data-size cutoff for that auto-choice.
- **`MIN_DISTINCT_VERBAL = 3`** — a cell needs at least 3 different
  confidence values from word-based answers, or it's excluded.
- **`QUADRANT_THRESHOLD = 0.5`** — the cutoff deciding "high" vs. "low"
  confidence when sorting questions into the "hopeful/suppressed
  confidence" quadrant groups.
- **`HLR_METHOD = "auto"`** — which statistical technique runs the big
  pooled model combining results across all 30 cells; "auto" lets the code
  decide.

---

## Checkpoint / IO settings

- **`RESUME = True`** — if a run gets interrupted, the next run picks up
  where it left off.
- **`CHECKPOINT_EVERY = 50`** — saves progress to disk every 50 questions.
- **`SAVE_RAW_TEXT = True`** — keeps the model's full original answer text,
  not just right/wrong verdicts.
- **`SAVE_ACTIVATIONS = True`** — keeps the internal snapshot numbers saved
  to disk.
- **`COMPRESS_ACTIVATIONS = True`** — compresses those snapshot files to
  save disk space.
- **`JSONL_ENSURE_ASCII = False`** — non-English characters get saved as-is
  in output files, rather than converted into escape codes.

---

## Figures settings

- **`FIG_DPI = 200`** — sharpness/resolution of exported chart images.
- **`FIG_FORMATS = ("png", "pdf")`** — save every chart in both formats.
- **`FIG_STYLE = "paper"`** — clean academic-paper look, not dark-mode.
- **`FIG_WIDTH = 7.2`** — inches, sized for a standard two-column paper.
- **`LATEX_TABLES = True`** — also export result tables in LaTeX format.

---

## The 14 Stages — plain English

Order matters — each stage feeds the next.

1. **`data`** — Build the question set (6 categories) and split into
   train/calibrate/test buckets.
2. **`pilot`** — Quick 100-question test per model to check the difficulty
   is reasonable (25%–80% accuracy) before committing to the full run.
3. **`verbal`** — Ask the model to state its confidence 3 different ways:
   a percentage, a word ("certain"/"unsure"), and a bet (answer or pass).
4. **`forced`** — Anywhere the model "passed," force it to answer anyway,
   to check if it actually knew but hedged.
5. **`sample`** — Ask each question 10 times with some randomness; same
   answer every time = confident, scattered answers = guessing.
6. **`extract`** — Grab snapshots of the model's internal numbers at 5
   different depths while it processes a question.
7. **`grade`** — Mark all collected answers right or wrong.
8. **`entropy`** — Turn the 10 repeated answers (step 5) into one
   consistency score.
9. **`probe`** — Test whether the internal snapshots (step 6) can predict
   correctness — does the model "secretly know" even if it doesn't say so?
10. **`calibrate`** — Adjust all three confidence signals so they're
    genuinely honest probabilities, comparable to each other.
11. **`stats`** — Run all the statistical comparisons and summaries.
12. **`figures`** — Generate the actual charts.
13. **`tables`** — Export results as CSV/formatted tables for the paper.
14. **`report`** — Final write-up: settings used, which quality gates
    passed/failed, full traceability record.

**One-line summary:** *build the test → get three kinds of "confidence" out
of the model (talking, repeating itself, internal signals) → grade
everything → compare the three → turn it into a report.*

---

## Cell 4 — Paths, Provenance, Checkpointed IO

**Big picture:** this cell doesn't run any AI models — it's the
"infrastructure" cell. It sets up where files get saved, writes a "receipt"
recording exactly what produced this run, locks down randomness so results
are reproducible, and builds the save/resume system so a long run survives
crashes.

### 1. Folder setup (`PLATFORM_PATHS`, `_resolve_paths`)

A lookup table of "if I'm on Kaggle, save here; if I'm on Colab, save
there," etc. `_resolve_paths()` then creates a whole tree of subfolders
inside the run's output folder — separate folders for raw data, internal
snapshots, figures, tables, logs, and so on — so the rest of the code can
just reference `PATHS["figures"]` instead of typing out paths everywhere.

⚠️ This calls `cfg.resolved_platform()` — the method flagged earlier as
possibly buggy. This is the cell where that bug would actually surface, if
it exists.

### 2. Telling HuggingFace where to save downloads

Sets environment variables so the HuggingFace library (which downloads the
AI models) saves everything into the run's designated cache folder instead
of its default location. Also applies the API token and "offline mode"
flag from the config, if set.

### 3. The "receipt" for this run (`code_sha`, `build_provenance`)

Implements the traceability requirement from the plan (§14.4).

- **`code_sha()`** — grabs a short fingerprint of the current code version
  from Git. Falls back to `"nogit"` if Git isn't available.
- **`build_provenance()`** — builds one dictionary: run name, config
  fingerprint, seed, code version, platform/GPU info, start time, and exact
  library versions. Saved to disk (`provenance.json`) alongside a full copy
  of the settings (`config.json`) — so any result can later be traced back
  to exactly what produced it.

### 4. Making randomness repeatable (`set_all_seeds`)

Several libraries (Python's built-in randomness, NumPy, PyTorch, the GPU)
each keep their own separate randomness generator. This function locks all
of them to the same seed number, so re-running the same code twice gives
identical results instead of silently different ones.

### 5. Basic file read/write helpers

- **`jsonl_read` / `jsonl_append`** — JSONL = one JSON record per line.
  `jsonl_read` loads all lines back in, quietly skipping any broken line
  (in case a previous session got killed mid-write). `jsonl_append` adds
  new records and force-writes them to disk immediately (`fsync`) rather
  than just holding them in memory — per the comment in the code: *"a
  checkpoint that is not on disk is not a checkpoint."*
- **`json_write` / `json_read`** — same idea, but for a single whole JSON
  file (e.g. the provenance receipt), not line-by-line.

### 6. The `Checkpoint` class — the actual "resume" mechanism

This is the real machinery behind `RESUME = True` and
`CHECKPOINT_EVERY = 50` from the config. Works like a to-do list tracker:

- On startup, reads whatever's already saved from a previous run and
  remembers which questions are already done (`self.done`).
- `.has(...)` — quickly check "have I already processed this one?"
- `.add(...)` — record a new result; buffers in memory and writes to disk
  every 50 records (`flush_every`) instead of one at a time (more
  efficient).
- `.flush()` — force-write whatever's waiting, even below 50.

**Why it matters:** if a Kaggle session dies at question 743/1000, the next
run reads the file, sees 1–743 are already done, and resumes at 744 —
instead of redoing (or losing) earlier work.

### 7. The `RunLog` class — a running diary of what happened

Every important event (a stage starting, a gate passing/failing, etc.)
gets written to a log file **and** printed to screen with a timestamp. This
becomes the actual data behind the plan's §17.2 "run log" table — a record
of what actually happened, vs. what was planned. Right after being defined,
it's immediately used to log `"session_start"`.

### 8. Small helper functions at the bottom

- **`cell_id(model, tier)`** — glues a model name + category name into one
  label, e.g. `"qwen2.5-7b-instruct__R1"` — a unique ID for one of the 30
  cells.
- **`free_cuda()`** — the actual GPU memory cleanup function (what
  `EMPTY_CACHE_EVERY_BATCHES` from the config triggers). Clears unused
  memory so the next model has room to load.
- **`vram_report()`** — checks how much GPU memory is currently
  used/reserved/peaked — useful for debugging out-of-memory issues.

**One-line summary of Cell 4:** *set up organized output folders, write a
"receipt" of exactly what settings/code produced this run, lock down
randomness for reproducibility, and build the save-and-resume system so a
long, expensive run can survive crashes without losing progress.*

---

## Cell 5 — Building the question bank (PLAN §3)

**Big picture:** downloads and standardizes questions from 4 real datasets
(PopQA, SimpleQA, GSM8K, MATH) into the 6 tiers, then splits them into
train/calibration/test — this is the actual `data` stage from the pipeline.

- **`hf_load` / `load_with_fallbacks`** — downloads a tier's dataset from
  HuggingFace; if the main source fails, tries backup sources
  (`TIER_FALLBACKS`) before giving up.
- **Answer-parsing helpers** — each dataset stores its "correct answer"
  differently, so each tier gets its own extractor:
  - `parse_popqa_answers` — pulls out *all* acceptable answer variants
    (nicknames/aliases) for R1/R2, so grading needs zero fuzzy matching.
  - `gsm8k_answer` — grabs the final number after the `####` marker (C1).
  - `math_answer` — pulls the answer out of a `\boxed{...}` LaTeX command,
    carefully handling nested braces (C2/C3).
  - `math_level` — reads the difficulty level number off a MATH problem.
- **`build_tier_rows`** — turns each tier's raw data into one standard
  shape (`qid, tier, question, answers, meta`). R1/R2 filter by a
  popularity score (top 20% = R1/popular, bottom 20% = R2/obscure); C2/C3
  filter by difficulty level. Rows always get shuffled at the end instead
  of just taking "the first N," to avoid hidden ordering bias.
- **`assign_splits`** — shuffles and divides each tier's questions into
  60% train / 20% calibration / 20% test, matching the config.
- **`build_question_bank`** — the orchestrator:
  - Reuses a previously-built bank from disk if the settings match exactly
    (same tiers/counts/seed/splits) — avoids rebuilding on every session.
  - Otherwise builds each tier fresh, caps it at `N_PER_CELL` (1000)
    questions, splits it, and — importantly — draws the pilot (100) and
    agreement-check (100) subsets **only from the train split**, never
    from calibration/test, to avoid data leakage.
  - Saves the finished bank plus a manifest (settings + provenance) to
    disk, so future runs can check "can I reuse this?"

**One-line summary:** *pull questions from 4 datasets into 6 standardized
tiers, randomly shuffle, split into train/cal/test, carve out pilot +
agreement subsets from train only, and cache it all to disk for reuse.*

---

## Cell 6 — Building the prompts (PLAN §4, §4.1)

**Big picture:** writes the exact wording sent to the model for every
question style (A/B/C/forced/sample), and forces a strict reply format so
answers can be auto-parsed later with simple pattern matching instead of
needing another AI to interpret free text.

- **`ANSWER_STYLE`** — tells the model what shape its answer should take,
  matched to the tier (`entity` for PopQA, `numeric` for GSM8K, `latex`
  for MATH, etc.).
- **`BUCKETS` / `BUCKET_ORDINAL`** — the 5 words for Format B
  (`CERTAIN` → `NO_IDEA`). The ordinal ranking is only for basic sanity
  checks — the real word→probability mapping gets fit empirically later
  (how often "CERTAIN" was actually right), not hand-assigned here.
- **`BET_GAIN, BET_LOSS = 1, -2`** — Format C's betting payoff: correct
  +1, wrong −2, pass 0. Deliberately unbalanced so passing only becomes
  the smart move once the model's true chance of being right drops below
  ~67% — turns "answer or pass" into a genuine confidence signal.
- **`instruction()`** — builds the actual instruction text per variant:
  - `A` — answer + confidence as a 0–100 number
  - `B` — answer + confidence as one of the 5 words
  - `C` — answer-or-pass with the betting payoff explained
  - `FORCED` — must answer, no passing/hedging allowed (the stage-4 follow-up)
  - `SAMPLE` / `EXTRACT` — just answer, no confidence talk (confidence is
    measured differently for these — via repetition or internal snapshots)
- **`FEWSHOT_POOL`** — 4 worked examples per variant, used only for the
  base model (which isn't instruction-tuned and needs the format
  demonstrated, not just described).
- **`build_prompt()`** — assembles the final prompt: for instruct models,
  builds a proper chat message and lets the tokenizer's chat template wrap
  it correctly; for the base model, manually stitches together the
  instructions + few-shot examples + the real question as plain text
  (since base models just continue text patterns, no chat format).

**One-line summary:** *defines the exact question-asking scripts for every
confidence-eliciting style, with a rigid output format for easy parsing,
and handles prompting instruct models (chat template) vs. the base model
(few-shot text completion) differently.*

---

## Open questions / things to check next

- [ ] Where is `PLATFORM` defined? (possible bug in `resolved_platform()`,
      now confirmed to matter in Cell 4's `_resolve_paths()` too — worth
      actually running this cell to see if it errors)
- [ ] Where does `purge_weights` actually get called, and how does it know
      a model's "final pass" is really final?
- [ ] What does `plan_model_batches` do? (referenced in a comment near
      `MODEL_REPLICAS`)
- [ ] What do `DEVICES` and `HAS_MATH_VERIFY` (referenced in
      `build_provenance`) come from — must be set in an earlier cell (Cell
      1 or 2, not yet seen)
- [ ] Where are `TIER_SPECS` and `TIER_FALLBACKS` actually defined? (Cell 5
      uses both heavily — likely in an earlier cell alongside `MODEL_SPECS`)
