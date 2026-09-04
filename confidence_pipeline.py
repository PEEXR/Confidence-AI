# %% [markdown]
# # Genuine vs. Hopeful Confidence in LLMs — full pipeline
#
# Single-notebook implementation of `PLAN.md` v3 (six-tier retrieval→reasoning
# ladder × Qwen2.5 model ladder × three confidence signals).
#
# **Runs on:** molab (1× RTX Pro 6000 Blackwell, 96 GB — recommended) or
# Kaggle (2× T4 — small models only, see the compute ledger at the end).
#
# **Everything you can turn is in `CFG` (cell 3).** Nothing below cell 3 needs
# editing for a normal run.
#
# ### Design decisions baked in
# | Decision | Choice | Why |
# |---|---|---|
# | Answer grading | Free-response + deterministic graders | No LLM judge. GSM8K numeric, MATH sympy, PopQA alias-list, SimpleQA→local NLI |
# | Output format | Rigid `KEY: VALUE` lines | Far higher compliance at 0.5B / base than JSON; stored as JSON on disk |
# | Precision | BF16 on Blackwell, FP16 on T4 | PLAN §9.1 clean activations; retires the T4 FP16 NaN risk on molab |
# | Resume | Idempotent, keyed by `(qid, variant)` | PLAN §10 — "a job that cannot resume is not a measurement" |
#
# ### Marimo / reactivity
# Written marimo-safe: no variable is defined in two cells, and every heavy
# stage lives inside a function gated on `CFG.STAGES`. Because every stage is
# **idempotent and resumable**, a reactive re-run costs seconds — it re-reads
# the checkpoints and skips completed work. Convert with:
# `marimo convert confidence_pipeline.ipynb -o confidence_pipeline.py`

# %%
# ============================================================================
# CELL 1 — bootstrap: dependencies, imports, platform + device detection
# ============================================================================
import importlib
import importlib.metadata
import importlib.util
import subprocess
import sys


def _pip(*pkgs):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-input", *pkgs],
        check=False,
    )


_REQUIRED = {
    "torch": "torch",
    "transformers": "transformers>=4.44",
    "datasets": "datasets>=2.20",
    "accelerate": "accelerate",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "statsmodels": "statsmodels",
    "sympy": "sympy",
    "tqdm": "tqdm",
}
_MISSING = [spec for mod, spec in _REQUIRED.items() if importlib.util.find_spec(mod) is None]
if _MISSING:
    _pip(*_MISSING)

# Optional: HuggingFace's robust MATH equivalence checker. Falls back to a
# sympy/normalisation grader if unavailable — never a hard dependency.
if importlib.util.find_spec("math_verify") is None:
    _pip("math-verify")

import gc
import hashlib
import json
import os
import platform as py_platform
import random
import re
import shutil
import threading
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.stats as sps
import torch
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HAS_MATH_VERIFY = importlib.util.find_spec("math_verify") is not None


def detect_platform() -> str:
    """kaggle | molab | colab | local — from filesystem + env markers."""
    if Path("/kaggle/working").exists():
        return "kaggle"
    if os.environ.get("MARIMO_MOLAB") or Path("/molab").exists():
        return "molab"
    try:
        has_marimo = "MARIMO_ROOT" in os.environ or importlib.util.find_spec("marimo") is not None
    except (ModuleNotFoundError, ValueError):
        has_marimo = False
    if has_marimo:
        # marimo is present; molab is the hosted flavour. Treat as molab only
        # when there is a GPU big enough to be the Blackwell box.
        if torch.cuda.is_available():
            try:
                if torch.cuda.get_device_properties(0).total_memory > 60e9:
                    return "molab"
            except Exception:
                pass
    try:
        if importlib.util.find_spec("google.colab") is not None:
            return "colab"
    except (ModuleNotFoundError, ValueError):
        pass
    return "local"


def detect_devices() -> dict:
    """Enumerate CUDA devices with the facts the config actually branches on."""
    info = {
        "cuda": torch.cuda.is_available(),
        "n_gpu": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
        "total_vram_gb": 0.0,
        "max_vram_gb": 0.0,
        "bf16": False,
        "torch": torch.__version__,
        "python": py_platform.python_version(),
    }
    for i in range(info["n_gpu"]):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1e9
        info["gpus"].append(
            {"index": i, "name": p.name, "vram_gb": round(gb, 2), "capability": f"{p.major}.{p.minor}"}
        )
        info["total_vram_gb"] += gb
        info["max_vram_gb"] = max(info["max_vram_gb"], gb)
    if info["cuda"]:
        # bf16 needs Ampere (sm80) or newer. Turing (T4, sm75) does not have it.
        info["bf16"] = bool(torch.cuda.is_bf16_supported())
    info["total_vram_gb"] = round(info["total_vram_gb"], 2)
    info["max_vram_gb"] = round(info["max_vram_gb"], 2)
    return info


PLATFORM = detect_platform()
DEVICES = detect_devices()

print(f"platform      : {PLATFORM}")
print(f"python/torch  : {DEVICES['python']} / {DEVICES['torch']}")
print(f"cuda devices  : {DEVICES['n_gpu']}")
for _g in DEVICES["gpus"]:
    print(f"  [{_g['index']}] {_g['name']}  {_g['vram_gb']} GB  sm_{_g['capability'].replace('.', '')}")
print(f"bf16 native   : {DEVICES['bf16']}")
print(f"math_verify   : {HAS_MATH_VERIFY}")

# %%
# ============================================================================
# CELL 2 — TIER + MODEL registries
# (referenced by CFG; edit here only to add a new dataset or model)
# ============================================================================

# ---------------------------------------------------------------- tiers ----
# PLAN §3: six-tier retrieval→reasoning ladder. `filter_fn` is applied to the
# raw HF rows; `max_new_tokens` and `cot` are the per-tier compute knobs that
# dominate the budget (see the ledger cell).
TIER_SPECS: dict[str, dict] = {
    "R1": dict(
        label="PopQA (top popularity quintile)",
        hf_id="akariasai/PopQA",
        hf_config=None,
        hf_split="test",
        family="retrieval",
        answer_form="entity",
        difficulty="popularity: top quintile",
        max_new_tokens=32,
        cot=False,
    ),
    "R2": dict(
        label="PopQA (bottom popularity quintile)",
        hf_id="akariasai/PopQA",
        hf_config=None,
        hf_split="test",
        family="retrieval",
        answer_form="entity",
        difficulty="popularity: bottom quintile",
        max_new_tokens=32,
        cot=False,
    ),
    "R3": dict(
        label="SimpleQA (adversarial retrieval)",
        hf_id="basicv8vc/SimpleQA",
        hf_config=None,
        hf_split="test",
        family="retrieval",
        answer_form="short",
        difficulty="dataset design",
        max_new_tokens=32,
        cot=False,
    ),
    "C1": dict(
        label="GSM8K",
        hf_id="openai/gsm8k",
        hf_config="main",
        hf_split="test",
        family="reasoning",
        answer_form="numeric",
        difficulty="—",
        max_new_tokens=256,
        cot=True,
    ),
    "C2": dict(
        label="MATH levels 1–2",
        hf_id="qwedsacf/competition_math",
        hf_config=None,
        hf_split="train",
        family="reasoning",
        answer_form="latex",
        difficulty="built-in level 1–2",
        max_new_tokens=320,
        cot=True,
    ),
    "C3": dict(
        label="MATH levels 4–5",
        hf_id="qwedsacf/competition_math",
        hf_config=None,
        hf_split="train",
        family="reasoning",
        answer_form="latex",
        difficulty="built-in level 4–5",
        max_new_tokens=448,
        cot=True,
    ),
}

# Fallback repo ids tried in order if the primary fails (HF datasets that lost
# script support, got renamed, or are gated).
TIER_FALLBACKS: dict[str, list[tuple[str, str | None, str]]] = {
    "C2": [("EleutherAI/hendrycks_math", "algebra", "train"), ("nlile/hendrycks-MATH-benchmark", None, "train")],
    "C3": [("EleutherAI/hendrycks_math", "algebra", "train"), ("nlile/hendrycks-MATH-benchmark", None, "train")],
    "R3": [("lighteval/SimpleQA", None, "test")],
}

# --------------------------------------------------------------- models ----
# PLAN §9. `layers` is the transformer block count; percentile→layer index uses
# it directly (index 0 = embedding output, index L = final block output).
MODEL_SPECS: dict[str, dict] = {
    "qwen2.5-0.5b-instruct": dict(
        hf_id="Qwen/Qwen2.5-0.5B-Instruct", params_b=0.49, layers=24, hidden=896,
        chat=True, rung="ladder",
    ),
    "qwen2.5-1.5b-instruct": dict(
        hf_id="Qwen/Qwen2.5-1.5B-Instruct", params_b=1.54, layers=28, hidden=1536,
        chat=True, rung="ladder",
    ),
    "qwen2.5-3b-instruct": dict(
        hf_id="Qwen/Qwen2.5-3B-Instruct", params_b=3.09, layers=36, hidden=2048,
        chat=True, rung="ladder",
    ),
    "qwen2.5-7b-instruct": dict(
        hf_id="Qwen/Qwen2.5-7B-Instruct", params_b=7.62, layers=28, hidden=3584,
        chat=True, rung="ladder",
    ),
    "qwen2.5-7b-base": dict(
        hf_id="Qwen/Qwen2.5-7B", params_b=7.62, layers=28, hidden=3584,
        chat=False, rung="h3-comparison",
    ),
}

# %%
# ============================================================================
# CELL 3 — >>> CONFIG <<<  every knob lives here
# ============================================================================


@dataclass
class Config:
    # ---------------------------------------------------------- identity --
    RUN_NAME: str = "prod500"
    SEED: int = 20260813
    NOTES: str = "pre-registered run per PLAN.md v3"

    # ---------------------------------------------------------- platform --
    PLATFORM: str = "auto"              # auto | molab | kaggle | colab | local
    OUTPUT_ROOT: str = ""               # "" = auto per platform
    HF_CACHE: str = ""                  # "" = auto (kept OFF the output volume)
    HF_TOKEN: str = ""                  # or set env HF_TOKEN
    HF_OFFLINE: bool = False

    # --------------------------------------------------- stage switches ---
    # Drop any name to skip that stage entirely. Order is the execution order.
    STAGES: tuple[str, ...] = (
        "data",       # build the six-tier question bank + splits
        "pilot",      # 100-q/cell accuracy pilot -> 25–80% band gate (PLAN §3)
        "verbal",     # Signal 1: formats A/B/C            (PLAN §4)
        "forced",     # forced-answer companion on Format C passes (PLAN §4.1)
        "sample",     # Signal 2: N=10 sampling            (PLAN §5)
        "extract",    # Signal 3: 5-percentile hook extraction (PLAN §6)
        "grade",      # deterministic grading of everything (PLAN §7)
        "entropy",    # semantic entropy from the samples  (PLAN §5)
        "probe",      # logistic probes + Gate 3           (PLAN §6)
        "calibrate",  # per-signal isotonic/Platt          (PLAN §8)
        "stats",      # Murphy, ECE/Brier, Spearman, HLR, index, quadrants
        "figures",    # Figures 1–4 + supplementary
        "tables",     # CSV + LaTeX exports
        "report",     # provenance, ledger, gate verdicts
    )

    # ------------------------------------------------------ grid subset ---
    # ONLY_* wins over SKIP_* when non-empty. Use these to split a long run
    # across sessions, or to re-run a single cell.
    ONLY_MODELS: tuple[str, ...] = ()
    SKIP_MODELS: tuple[str, ...] = ()
    ONLY_TIERS: tuple[str, ...] = ()
    SKIP_TIERS: tuple[str, ...] = ()

    # ------------------------------------------------- question budgets ---
    N_PILOT: int = 100                  # PLAN §3 pilot size
    N_PER_CELL: int = 500               # committed-cell size (15,000 total items across 30 cells)
    N_AGREEMENT: int = 100              # H0 / Gate 2 subset (PLAN §4)
    N_MANUAL_CHECK: int = 50            # Gate 1 hand-verification sample size
    SPLIT_FRACTIONS: tuple[float, float, float] = (0.6, 0.2, 0.2)  # train/cal/test

    # ------------------------------------------------------- generation ---
    DTYPE: str = "auto"                 # auto | bfloat16 | float16 | float32
    ATTN_IMPL: str = "sdpa"             # sdpa is the safe choice on both T4 and Blackwell
    BATCH_SIZE: int = 0                 # 0 = auto-size from free VRAM
    BATCH_SIZE_CAP: int = 256
    GREEDY_TEMPERATURE: float = 0.0     # formats A/B/C + extraction pass
    SAMPLE_TEMPERATURE: float = 0.8     # PLAN §5: must be in 0.7–1.0
    SAMPLE_TOP_P: float = 0.95
    N_SAMPLES: int = 10                 # PLAN §5 N=10
    # Entropy is normalised by log(N_SAMPLES), never by the number of samples
    # that happened to parse; a question below ENTROPY_MIN_VALID is NaN, not 1.0.
    ENTROPY_MIN_VALID: int = 8          # < this many parsed samples -> NaN
    ENTROPY_LP_WEIGHTED: bool = True    # Rao-Blackwellise cluster mass by sequence log-probs
    N_FEWSHOT_BASE: int = 4             # few-shot exemplars for the base model
    STOP_ON_DOUBLE_NEWLINE: bool = False

    # ------------------------------------------- model execution policy ---
    MODEL_EXEC: str = "sequential"      # sequential | resident | concurrent
    MAX_CONCURRENT_MODELS: int = 2      # only used when MODEL_EXEC == "concurrent"
    CONCURRENT_MAX_PARAMS_B: float = 4.0  # models bigger than this never run concurrently
    # Replicas of the SAME weights, each taking a disjoint slice of the tier
    # ladder. Read the note in `plan_model_batches` before raising this above 1:
    # decode is memory-bandwidth bound, so N replicas each re-read their own
    # copy of the weights — a single replica at N x the batch size is strictly
    # better unless you are CPU-bound on the generate loop.
    MODEL_REPLICAS: int = 1
    # never       - keep every snapshot (best on molab; disk is effectively free)
    # after_model - delete a model's snapshot once it finishes its FINAL pass
    #               (never after the pilot pass, or it would re-download)
    # after_run   - delete the whole hub cache once, at the very end
    PURGE_WEIGHTS: str = "never"        # never | after_model | after_run
    EMPTY_CACHE_EVERY_BATCHES: int = 4

    # ------------------------------------------------------- band gate ----
    # PLAN §3's 25-80% band is a legitimate base-rate control, but DELETING
    # out-of-band cells removed all of R2, all of R3, all of 0.5B and 5/6 of
    # C3 — three quarters of the grid — on the strength of a noisy n=100 pilot
    # (AUDIT finding 9). The band is now recorded as a covariate (`in_band`)
    # and the control is applied statistically, by difficulty matching and by
    # the GLMM's cell random intercepts. Set COMMIT_CELLS_OUTSIDE_BAND=False
    # to restore the old deleting behaviour.
    ACCURACY_BAND: tuple[float, float] = (0.25, 0.80)   # PLAN §3 — reported, not enforced
    COMMIT_CELLS_OUTSIDE_BAND: bool = True              # False = delete out-of-band cells (pre-audit)

    # ------------------------------------------------------- probe (§6) ---
    PERCENTILES: tuple[int, ...] = (0, 25, 50, 75, 100)
    # PLAN §6·5 recommends the ENTROPY label. Training on `correct` makes the
    # probe a second supervised accuracy predictor, so the "three-signal
    # comparison" races one unsupervised signal against two supervised ones
    # (AUDIT finding 4). "correct" remains available as a documented
    # supervised reference mode — it is no longer the default.
    PROBE_LABEL: str = "entropy"        # entropy | correct  (PLAN §6·5)
    PROBE_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    PROBE_C_SELECT: bool = True         # pick C on the calibration split, never hardcode
    PROBE_MAX_ITER: int = 2000
    PROBE_STORE_DTYPE: str = "float32"  # PLAN §16 standing risk 1
    AUROC_GATE: float = 0.65            # Gate 3
    GATE3_ENFORCE: bool = True          # block probe fitting on cells with dirty activations
    # The embedding of a fixed prompt-ending token carries no question
    # information, so p0 MUST score ~0.50. The pre-audit tap scored 0.717
    # there, which is what exposed it as reading a generated token. This is now
    # a blocking control, not a diagnostic.
    P0_NEG_CONTROL_TOL: float = 0.10    # |AUROC(p0) - 0.5| above this fails Gate 3
    LABEL_SHUFFLE_REPEATS: int = 20     # null distribution size
    SURFACE_BASELINE: bool = True       # TF-IDF prompt-only control (PLAN §14.1)

    # ------------------------------------------------------- grading -----
    NUMERIC_TOLERANCE: float = 1e-6
    USE_NLI_FALLBACK: bool = True
    NLI_MODEL: str = "microsoft/deberta-large-mnli"
    NLI_ENTAIL_THRESHOLD: float = 0.70
    NLI_BATCH_SIZE: int = 64
    STRIP_ARTICLES: bool = True

    # ---------------------------------------------------------- gates ----
    GATE1_AGREEMENT: float = 0.95       # grading sanity (PLAN §16)
    GATE2_SPEARMAN: float = 0.60        # format agreement / H0
    GATE4_REQUIRE_BASE_ELICITATION: bool = True
    BASE_ELICITATION_MIN_PARSE_RATE: float = 0.50   # E5 usability bar

    # ------------------------------------------------------ statistics ---
    N_BOOTSTRAP: int = 2000
    BOOTSTRAP_CI: float = 0.95
    ECE_BINS: int = 15
    MURPHY_BINS: int = 10
    CALIBRATOR: str = "auto"            # auto | isotonic | platt
    ISOTONIC_MIN_N: int = 200           # below this, auto falls back to Platt
    MIN_DISTINCT_VERBAL: int = 3        # PLAN §8·6 verbal pre-flight
    QUADRANT_THRESHOLD: float = 0.5     # split point on calibrated scores
    # One variant defines ground-truth correctness for the whole comparison.
    # Accuracy differs by up to 12 points across A/B/C/SAMPLE/EXTRACT, so
    # scoring the verbal signal against B's correctness while the probe trains
    # on EXTRACT's correctness compares three signals to three different
    # targets (AUDIT finding 4). EXTRACT is the plain answering pass with no
    # confidence elicitation in context, and it is the pass the activations
    # were tapped from.
    GROUND_TRUTH_VARIANT: str = "EXTRACT"   # EXTRACT | FORCED | canonical_verbal
    HLR_METHOD: str = "auto"            # auto | bayes_mixed | cluster_robust

    # ------------------------------------------------- checkpoint / io ---
    RESUME: bool = True
    CHECKPOINT_EVERY: int = 50          # records between flushes
    SAVE_RAW_TEXT: bool = True          # keep full generations, not just parses
    SAVE_ACTIVATIONS: bool = True
    COMPRESS_ACTIVATIONS: bool = True   # npz-compressed shards
    JSONL_ENSURE_ASCII: bool = False

    # -------------------------------------------------------- figures ----
    FIG_DPI: int = 200
    FIG_FORMATS: tuple[str, ...] = ("png", "pdf")
    FIG_STYLE: str = "paper"            # paper | dark
    FIG_WIDTH: float = 7.2              # inches; two-column figure width
    LATEX_TABLES: bool = True

    # --------------------------------------------------------- derived ---
    def resolved_platform(self) -> str:
        return PLATFORM if self.PLATFORM == "auto" else self.PLATFORM

    def resolved_dtype(self) -> "torch.dtype":
        if self.DTYPE != "auto":
            return getattr(torch, self.DTYPE)
        if not DEVICES["cuda"]:
            return torch.float32
        return torch.bfloat16 if DEVICES["bf16"] else torch.float16

    def active_models(self) -> list[str]:
        names = list(MODEL_SPECS)
        if self.ONLY_MODELS:
            names = [n for n in names if n in self.ONLY_MODELS]
        return [n for n in names if n not in self.SKIP_MODELS]

    def active_tiers(self) -> list[str]:
        names = list(TIER_SPECS)
        if self.ONLY_TIERS:
            names = [n for n in names if n in self.ONLY_TIERS]
        return [n for n in names if n not in self.SKIP_TIERS]

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


_BASE_CFG = Config()

# --------------------------------------------------------------------------
# SMOKE — set False for the real run.
#
# Exercises every stage end-to-end on a handful of questions. At this size the
# 60/20/20 split is degenerate (3/1/1 per cell), so the probe and calibration
# stages will legitimately skip: they need >=20 train and >=10 calibration
# rows. That is the expected smoke outcome, not a failure. The band gate is
# also bypassed, because a 5-question pilot cannot meaningfully land in the
# 25-80% accuracy band.
# --------------------------------------------------------------------------
SMOKE = False

CFG = replace(
    _BASE_CFG,
    RUN_NAME="smoke", N_PILOT=5, N_PER_CELL=5, N_AGREEMENT=5, N_MANUAL_CHECK=5,
    N_SAMPLES=3, N_BOOTSTRAP=200,
    ONLY_MODELS=("qwen2.5-0.5b-instruct",),
    COMMIT_CELLS_OUTSIDE_BAND=True,
    USE_NLI_FALLBACK=False,
) if SMOKE else _BASE_CFG

# --------------------------------------------------------------------------
# Other quick-start overrides — uncomment instead of editing the dataclass.
# --------------------------------------------------------------------------
# H0-only abort branch (PLAN §13 — publishable alone, no probe, no sampling):
# CFG = replace(CFG, RUN_NAME="h0_only",
#               STAGES=("data", "verbal", "grade", "stats", "figures", "tables", "report"))
#
# Session 1 of 2 — small models, leave the 7B pair for session 2:
# CFG = replace(CFG, RUN_NAME="s1", SKIP_MODELS=("qwen2.5-7b-instruct", "qwen2.5-7b-base"),
#               MODEL_EXEC="concurrent", MAX_CONCURRENT_MODELS=3)
#
# Session 2 of 2 — the 7B pair only (resumes shared question bank):
# CFG = replace(CFG, RUN_NAME="s1",
#               ONLY_MODELS=("qwen2.5-7b-instruct", "qwen2.5-7b-base"))

print(f"config hash   : {CFG.hash()}")
print(f"run name      : {CFG.RUN_NAME}")
print(f"dtype         : {CFG.resolved_dtype()}")
print(f"models        : {CFG.active_models()}")
print(f"tiers         : {CFG.active_tiers()}")
print(f"stages        : {CFG.STAGES}")
# %%
# ============================================================================
# CELL 4 — paths, provenance (X1), checkpointed JSONL io
# ============================================================================

PLATFORM_PATHS = {
    "kaggle": dict(out="/kaggle/working/confidence", cache="/kaggle/temp/hf"),
    "molab":  dict(out="./confidence_out",          cache="./hf_cache"),
    "colab":  dict(out="/content/confidence",       cache="/content/hf_cache"),
    "local":  dict(out="./confidence_out",          cache=""),
}


def _resolve_paths(cfg: Config) -> dict:
    d = PLATFORM_PATHS.get(cfg.resolved_platform(), PLATFORM_PATHS["local"])
    out = Path(cfg.OUTPUT_ROOT or d["out"]) / cfg.RUN_NAME
    cache = cfg.HF_CACHE or d["cache"]
    tree = {
        "root": out,
        "data": out / "data",
        "raw": out / "raw",
        "acts": out / "activations",
        "derived": out / "derived",
        "figures": out / "figures",
        "tables": out / "tables",
        "meta": out / "meta",
        "logs": out / "logs",
    }
    for p in tree.values():
        p.mkdir(parents=True, exist_ok=True)
    tree["hf_cache"] = Path(cache) if cache else None
    return tree


PATHS = _resolve_paths(CFG)

if PATHS["hf_cache"]:
    PATHS["hf_cache"].mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(PATHS["hf_cache"])
    os.environ["HF_DATASETS_CACHE"] = str(PATHS["hf_cache"] / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(PATHS["hf_cache"] / "transformers")
if CFG.HF_TOKEN:
    os.environ["HF_TOKEN"] = CFG.HF_TOKEN
if CFG.HF_OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def code_sha() -> str:
    """The commit this run's code came from.

    Every shipped run recorded `code_sha: "nogit"`, so no result is bound to a
    version of the pipeline (AUDIT finding 8). On Molab/Kaggle the notebook is
    uploaded without its repository, so a bare `git rev-parse` in the working
    directory finds nothing and the failure is silent. Three sources are tried,
    in order of trustworthiness, and an unbound run now says so loudly instead
    of writing a placeholder that reads like a value.
    """
    env = os.environ.get("CODE_SHA", "").strip()
    if env:
        return env[:40]
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for start in (here, Path.cwd()):
        for d in [start, *start.parents]:
            if (d / ".git").exists():
                try:
                    r = subprocess.run(["git", "-C", str(d), "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True, timeout=5)
                    if r.returncode == 0 and r.stdout.strip():
                        sha = r.stdout.strip()
                        dirty = subprocess.run(["git", "-C", str(d), "status", "--porcelain"],
                                               capture_output=True, text=True, timeout=5)
                        return sha + ("-dirty" if dirty.stdout.strip() else "")
                except Exception:                        # noqa: BLE001
                    pass
    warnings.warn(
        "code_sha is UNBOUND: no .git found and CODE_SHA is unset, so this run cannot be "
        "traced to a version of the pipeline. On Molab/Kaggle, set CODE_SHA in the environment "
        "before running (e.g. os.environ['CODE_SHA'] = '<sha from git rev-parse HEAD>').",
        RuntimeWarning, stacklevel=2)
    return "UNBOUND-nogit"


def build_provenance(cfg: Config) -> dict:
    """PLAN §14.4 / X1 — stamped onto every derived artefact."""
    return {
        "run_name": cfg.RUN_NAME,
        "config_hash": cfg.hash(),
        "seed": cfg.SEED,
        "code_sha": code_sha(),
        "platform": cfg.resolved_platform(),
        "devices": DEVICES,
        "dtype": str(cfg.resolved_dtype()),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "transformers": importlib.metadata.version("transformers"),
        "torch": torch.__version__,
        "math_verify": HAS_MATH_VERIFY,
    }


PROV = build_provenance(CFG)
(PATHS["meta"] / "provenance.json").write_text(json.dumps(PROV, indent=2, default=str))
(PATHS["meta"] / "config.json").write_text(json.dumps(asdict(CFG), indent=2, default=str))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_all_seeds(CFG.SEED)


# ------------------------------------------------------------------ io ----
def jsonl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # tolerate a torn final line from a killed session
    return out


def jsonl_append(path: Path, records: Sequence[dict], ensure_ascii: bool = False) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=ensure_ascii, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())     # a checkpoint that is not on disk is not a checkpoint


def json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def json_read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


class Checkpoint:
    """Resumable JSONL sink keyed by an arbitrary tuple (PLAN §10, §17.3).

    Contract: `done` holds every key already on disk; `add` buffers and flushes
    every `flush_every` records. Re-running a completed stage is a no-op that
    costs one file read — which is what makes reactive re-execution safe.
    """

    def __init__(self, path: Path, key_fields: Sequence[str], flush_every: int = 50, resume: bool = True):
        self.path = path
        self.key_fields = tuple(key_fields)
        self.flush_every = flush_every
        self._buf: list[dict] = []
        self.done: set[tuple] = set()
        if resume:
            for rec in jsonl_read(path):
                self.done.add(self._key(rec))
        elif path.exists():
            path.unlink()

    def _key(self, rec: dict) -> tuple:
        return tuple(rec.get(k) for k in self.key_fields)

    def has(self, **kw) -> bool:
        return tuple(kw.get(k) for k in self.key_fields) in self.done

    def add(self, rec: dict) -> None:
        self._buf.append(rec)
        self.done.add(self._key(rec))
        if len(self._buf) >= self.flush_every:
            self.flush()

    def extend(self, recs: Iterable[dict]) -> None:
        for r in recs:
            self.add(r)

    def flush(self) -> None:
        if self._buf:
            jsonl_append(self.path, self._buf, CFG.JSONL_ENSURE_ASCII)
            self._buf = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.flush()
        return False


class RunLog:
    """Append-only event log; also the source of the §17.2 run-log table."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def log(self, event: str, **fields) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **fields}
        with self._lock:
            jsonl_append(self.path, [rec])
        msg = " ".join(f"{k}={v}" for k, v in fields.items() if k != "detail")
        print(f"[{rec['ts'][11:19]}] {event:22s} {msg}")


LOG = RunLog(PATHS["logs"] / "events.jsonl")
LOG.log("session_start", platform=CFG.resolved_platform(), config_hash=CFG.hash(), root=str(PATHS["root"]))


def cell_id(model: str, tier: str) -> str:
    return f"{model}__{tier}"


def free_cuda() -> None:
    """The 'clear GPU memory' contract — called after every model finishes."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def vram_report() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        f"gpu{i}": {
            "alloc_gb": round(torch.cuda.memory_allocated(i) / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(i) / 1e9, 2),
            "peak_gb": round(torch.cuda.max_memory_allocated(i) / 1e9, 2),
        }
        for i in range(torch.cuda.device_count())
    }


# %%
# ============================================================================
# CELL 5 — dataset construction: six tiers, random sample, splits (PLAN §3)
# ============================================================================
from datasets import load_dataset  # noqa: E402


def hf_load(hf_id: str, hf_config: str | None, split: str):
    kwargs = {"split": split}
    if hf_config:
        return load_dataset(hf_id, hf_config, **kwargs)
    return load_dataset(hf_id, **kwargs)


def load_with_fallbacks(tier: str, spec: dict):
    attempts = [(spec["hf_id"], spec.get("hf_config"), spec["hf_split"])] + TIER_FALLBACKS.get(tier, [])
    errors = []
    for hf_id, cfgname, split in attempts:
        try:
            ds = hf_load(hf_id, cfgname, split)
            if hf_id != spec["hf_id"]:
                LOG.log("dataset_fallback", tier=tier, used=hf_id, primary=spec["hf_id"])
            return ds, hf_id
        except Exception as exc:                        # noqa: BLE001
            errors.append(f"{hf_id}({cfgname},{split}): {type(exc).__name__}: {exc}")
    raise RuntimeError(f"tier {tier}: all dataset sources failed:\n  " + "\n  ".join(errors))


def parse_popqa_answers(row: dict) -> list[str]:
    """PopQA ships `possible_answers` as a JSON-encoded list — the alias list is
    what makes this tier gradeable with zero fuzzy matching."""
    out: list[str] = []
    for fld in ("possible_answers", "o_aliases"):
        raw = row.get(fld)
        if raw is None:
            continue
        if isinstance(raw, list):
            out.extend(str(x) for x in raw)
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                out.extend(str(x) for x in parsed) if isinstance(parsed, list) else out.append(raw)
            except json.JSONDecodeError:
                out.append(raw)
    if row.get("obj"):
        out.append(str(row["obj"]))
    seen, uniq = set(), []
    for a in out:
        a = a.strip()
        if a and a.lower() not in seen:
            seen.add(a.lower())
            uniq.append(a)
    return uniq


def gsm8k_answer(raw: str) -> str:
    m = re.search(r"####\s*(.+)$", raw.strip())
    return (m.group(1) if m else raw).strip().replace(",", "")


def math_answer(solution: str) -> str | None:
    """Extract the content of the last \\boxed{...}, brace-balanced."""
    idx = solution.rfind("\\boxed")
    if idx < 0:
        m = re.search(r"\\fbox\{", solution)
        if not m:
            return None
        idx = m.start()
    i = solution.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(solution):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[i + 1 : j].strip()
        j += 1
    return None


def math_level(row: dict) -> int | None:
    lvl = str(row.get("level", ""))
    m = re.search(r"(\d)", lvl)
    return int(m.group(1)) if m else None


def build_tier_rows(tier: str, spec: dict, rng: np.random.Generator) -> list[dict]:
    """Return normalised question records for one tier, randomly sampled.

    Every record: {qid, tier, question, answers[list of acceptable], meta}
    """
    ds, source = load_with_fallbacks(tier, spec)
    rows: list[dict] = []

    if tier in ("R1", "R2"):
        pops = np.array([float(r) if r is not None else 0.0 for r in ds["s_pop"]])
        lo, hi = np.nanpercentile(pops, 20), np.nanpercentile(pops, 80)
        keep_hi = tier == "R1"
        for i, row in enumerate(ds):
            p = float(row.get("s_pop") or 0.0)
            if (keep_hi and p < hi) or ((not keep_hi) and p > lo):
                continue
            answers = parse_popqa_answers(row)
            if not answers or not row.get("question"):
                continue
            rows.append(dict(
                qid=f"{tier}-{row.get('id', i)}", tier=tier, question=row["question"].strip(),
                answers=answers,
                meta=dict(s_pop=p, prop=row.get("prop"), subj=row.get("subj"), source=source),
            ))

    elif tier == "R3":
        qkey = "problem" if "problem" in ds.column_names else "question"
        for i, row in enumerate(ds):
            q, a = row.get(qkey), row.get("answer")
            if not q or a is None:
                continue
            rows.append(dict(
                qid=f"{tier}-{i}", tier=tier, question=str(q).strip(), answers=[str(a).strip()],
                meta=dict(topic=(row.get("metadata") or {}).get("topic") if isinstance(row.get("metadata"), dict) else None,
                          source=source),
            ))

    elif tier == "C1":
        for i, row in enumerate(ds):
            ans = gsm8k_answer(row["answer"])
            rows.append(dict(
                qid=f"{tier}-{i}", tier=tier, question=row["question"].strip(), answers=[ans],
                meta=dict(source=source),
            ))

    elif tier in ("C2", "C3"):
        want = {1, 2} if tier == "C2" else {4, 5}
        for i, row in enumerate(ds):
            lvl = math_level(row)
            if lvl not in want:
                continue
            ans = math_answer(row.get("solution", "") or "")
            if not ans:
                continue
            rows.append(dict(
                qid=f"{tier}-{i}", tier=tier, question=str(row["problem"]).strip(), answers=[ans],
                meta=dict(level=lvl, subject=row.get("type"), source=source),
            ))

    else:
        raise KeyError(f"unknown tier {tier}")

    # PLAN/TASKS: never take the head of a sorted dataset — always a random sample.
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def assign_splits(rows: list[dict], fracs: tuple[float, float, float], rng: np.random.Generator) -> None:
    n = len(rows)
    idx = rng.permutation(n)
    n_tr = int(round(fracs[0] * n))
    n_cal = int(round(fracs[1] * n))
    for rank, i in enumerate(idx):
        rows[i]["split"] = "train" if rank < n_tr else ("calibration" if rank < n_tr + n_cal else "test")


def build_question_bank(cfg: Config) -> dict[str, list[dict]]:
    """Build (or resume) the shared six-tier question bank.

    The bank is model-independent, so it is built once and reused by every
    model — which is also what lets a run be split across sessions.
    """
    bank_path = PATHS["data"] / "question_bank.json"
    manifest_path = PATHS["data"] / "bank_manifest.json"
    manifest = json_read(manifest_path, {})
    want = {
        "tiers": cfg.active_tiers(), "n_per_cell": cfg.N_PER_CELL, "n_pilot": cfg.N_PILOT,
        "seed": cfg.SEED, "splits": list(cfg.SPLIT_FRACTIONS),
    }
    if cfg.RESUME and manifest.get("spec") == want and bank_path.exists():
        LOG.log("bank_reused", tiers=len(want["tiers"]))
        return json_read(bank_path)

    rng = np.random.default_rng(cfg.SEED)
    bank: dict[str, list[dict]] = {}
    for tier in cfg.active_tiers():
        spec = TIER_SPECS[tier]
        pool = build_tier_rows(tier, spec, rng)
        n_take = min(cfg.N_PER_CELL, len(pool))
        if n_take < cfg.N_PER_CELL:
            LOG.log("tier_short", tier=tier, available=len(pool), requested=cfg.N_PER_CELL)
        sel = pool[:n_take]
        assign_splits(sel, cfg.SPLIT_FRACTIONS, rng)
        # The pilot subset is drawn from the *train* split so the band gate
        # never touches calibration or test (PLAN §14.2).
        train_ids = [r["qid"] for r in sel if r["split"] == "train"]
        pilot_ids = set(train_ids[: min(cfg.N_PILOT, len(train_ids))])
        agree_ids = set(train_ids[: min(cfg.N_AGREEMENT, len(train_ids))])
        for r in sel:
            r["is_pilot"] = r["qid"] in pilot_ids
            r["is_agreement"] = r["qid"] in agree_ids
            r["family"] = spec["family"]
            r["answer_form"] = spec["answer_form"]
        bank[tier] = sel
        LOG.log("tier_built", tier=tier, n=len(sel), pool=len(pool),
                train=sum(r["split"] == "train" for r in sel),
                cal=sum(r["split"] == "calibration" for r in sel),
                test=sum(r["split"] == "test" for r in sel))

    json_write(bank_path, bank)
    json_write(manifest_path, {"spec": want, "provenance": PROV,
                               "counts": {t: len(v) for t, v in bank.items()}})
    return bank
# %%
# ============================================================================
# CELL 6 — prompts: formats A / B / C, forced-answer, sampling  (PLAN §4, §4.1)
# Rigid `KEY: VALUE` lines — parsed by regex, stored as JSON.
# ============================================================================

ANSWER_STYLE = {
    "entity":  "the entity name only, no sentence",
    "short":   "the shortest correct answer, no sentence",
    "numeric": "the final number only, no units, no commas",
    "latex":   "the final expression only, in simplest form, no \\boxed and no units",
}

BUCKETS: tuple[str, ...] = ("CERTAIN", "FAIRLY_CONFIDENT", "SOMEWHAT_UNSURE", "MOSTLY_GUESSING", "NO_IDEA")

# Fixed semantic values for the Format-B buckets, assigned from the plain
# meaning of each label and NEVER from data. AUDIT finding 2: mapping a bucket
# to the empirical accuracy of the answers placed in it makes "the model's
# stated confidence" a supervised P(correct) estimate wearing the model's
# label — which is why format B scored ECE 2.3e-17, identically zero by
# construction. These constants give B a label-free reading that can stand as
# a genuine verbalized axis; the empirical mapping survives alongside it as an
# explicitly supervised reference.
BUCKET_FIXED_VALUES: dict[str, float] = {
    "CERTAIN": 0.95,
    "FAIRLY_CONFIDENT": 0.75,
    "SOMEWHAT_UNSURE": 0.50,
    "MOSTLY_GUESSING": 0.30,
    "NO_IDEA": 0.10,
}
# Formats whose raw value is a genuine model utterance. Only these may become
# the canonical verbal axis. "B" is excluded by construction, in selection AND
# in the degenerate fallback.
LABEL_FREE_FORMATS: tuple[str, ...] = ("A", "Bfix", "C")
VERBAL_FORMATS: tuple[str, ...] = ("A", "B", "Bfix", "C")

# PLAN §4 format B: these are ordinal placeholders ONLY, used for sanity checks
# and never for scoring. The scored mapping is fit empirically in `stats`
# (PLAN §4 "bucket-to-probability mapping" / §14.3 manipulation check).
BUCKET_ORDINAL = {b: i for i, b in enumerate(reversed(BUCKETS))}

# Format C payoff (PLAN §4). Asymmetric so passing is rational below p≈2/3.
BET_GAIN, BET_LOSS = 1, -2


def reasoning_block(cot: bool) -> str:
    return "REASONING: <at most 3 short sentences of working>\n" if cot else ""


def instruction(variant: str, tier_spec: dict) -> str:
    style = ANSWER_STYLE[tier_spec["answer_form"]]
    cot = tier_spec["cot"]
    lines = reasoning_block(cot)
    if variant == "A":
        body = (
            "Answer the question, then state how confident you are that your answer is correct.\n\n"
            "Reply in EXACTLY this format and nothing else:\n"
            f"{lines}ANSWER: <{style}>\n"
            "CONFIDENCE: <an integer from 0 to 100>\n"
        )
    elif variant == "B":
        body = (
            "Answer the question, then state how confident you are that your answer is correct.\n\n"
            "Reply in EXACTLY this format and nothing else:\n"
            f"{lines}ANSWER: <{style}>\n"
            f"CONFIDENCE: <exactly one of: {', '.join(BUCKETS)}>\n"
        )
    elif variant == "C":
        body = (
            "You may either ANSWER or PASS.\n"
            f"A correct answer scores {BET_GAIN:+d}. An incorrect answer scores {BET_LOSS:+d}. "
            "A pass scores 0.\n"
            "Choose whichever maximises your expected score.\n\n"
            "Reply in EXACTLY this format and nothing else:\n"
            f"{lines}DECISION: <ANSWER or PASS>\n"
            f"ANSWER: <{style}, or NONE if you chose PASS>\n"
        )
    elif variant == "FORCED":
        body = (
            "Answer the question. You must give your single best answer — "
            "passing, refusing and saying you do not know are not permitted.\n\n"
            "Reply in EXACTLY this format and nothing else:\n"
            f"{lines}ANSWER: <{style}>\n"
        )
    elif variant in ("SAMPLE", "EXTRACT"):
        body = (
            "Answer the question.\n\n"
            "Reply in EXACTLY this format and nothing else:\n"
            f"{lines}ANSWER: <{style}>\n"
        )
    else:
        raise KeyError(variant)
    return body


# Few-shot exemplars for the base model, which has no chat template and needs
# the format demonstrated rather than instructed (PLAN §9 flag / E5).
FEWSHOT_POOL = {
    "A": [
        ("What is the capital of France?", "ANSWER: Paris\nCONFIDENCE: 99"),
        ("Who wrote the novel Beloved?", "ANSWER: Toni Morrison\nCONFIDENCE: 92"),
        ("In what year was the transistor invented?", "ANSWER: 1947\nCONFIDENCE: 78"),
        ("What is the surname of the mayor of Lisbon in 1954?", "ANSWER: Frade\nCONFIDENCE: 12"),
    ],
    "B": [
        ("What is the capital of France?", "ANSWER: Paris\nCONFIDENCE: CERTAIN"),
        ("Who wrote the novel Beloved?", "ANSWER: Toni Morrison\nCONFIDENCE: FAIRLY_CONFIDENT"),
        ("In what year was the transistor invented?", "ANSWER: 1947\nCONFIDENCE: SOMEWHAT_UNSURE"),
        ("What is the surname of the mayor of Lisbon in 1954?", "ANSWER: Frade\nCONFIDENCE: MOSTLY_GUESSING"),
    ],
    "C": [
        ("What is the capital of France?", "DECISION: ANSWER\nANSWER: Paris"),
        ("Who wrote the novel Beloved?", "DECISION: ANSWER\nANSWER: Toni Morrison"),
        ("In what year was the transistor invented?", "DECISION: ANSWER\nANSWER: 1947"),
        ("What is the surname of the mayor of Lisbon in 1954?", "DECISION: PASS\nANSWER: NONE"),
    ],
    "FORCED": [
        ("What is the capital of France?", "ANSWER: Paris"),
        ("Who wrote the novel Beloved?", "ANSWER: Toni Morrison"),
        ("In what year was the transistor invented?", "ANSWER: 1947"),
        ("What is the surname of the mayor of Lisbon in 1954?", "ANSWER: Frade"),
    ],
}
FEWSHOT_POOL["SAMPLE"] = FEWSHOT_POOL["FORCED"]
FEWSHOT_POOL["EXTRACT"] = FEWSHOT_POOL["FORCED"]


def build_prompt(variant: str, question: str, tier_spec: dict, model_spec: dict,
                 tokenizer, cfg: Config) -> str:
    """Return the fully-rendered prompt string for one (variant, question)."""
    instr = instruction(variant, tier_spec)
    if model_spec["chat"]:
        msgs = [
            {"role": "system", "content": "You are a precise assistant. You always reply in the exact requested format."},
            {"role": "user", "content": f"{instr}\nQuestion: {question}"},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Base model: instruction + few-shot completion, no chat template.
    shots = FEWSHOT_POOL[variant][: cfg.N_FEWSHOT_BASE]
    parts = [instr.rstrip(), ""]
    for q, a in shots:
        parts += [f"Question: {q}", a, ""]
    parts += [f"Question: {question}", ""]
    return "\n".join(parts)


# %%
# ============================================================================
# CELL 7 — parsers: rigid KEY: VALUE -> structured record
# Parse failure is a measured quantity, not an exception (PLAN §17.3).
# ============================================================================

KEY_RE_CACHE: dict[str, re.Pattern] = {}


def key_regex(key: str) -> re.Pattern:
    if key not in KEY_RE_CACHE:
        KEY_RE_CACHE[key] = re.compile(rf"^[\s\*\-#>]*{key}\s*[:：]\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
    return KEY_RE_CACHE[key]


# A few-shot prompted base model does not stop after one answer — it keeps
# emitting "Question: ... / ANSWER: ..." pairs forever. Everything after the
# first fabricated question belongs to a question we never asked, so it must be
# cut before parsing. Chat models never emit this marker, so this is a no-op
# for them and the rule can be applied uniformly.
_NEXT_Q_RE = re.compile(r"^\s*(?:Question|Q)\s*[:：]", re.IGNORECASE | re.MULTILINE)


def truncate_fewshot_continuation(text: str) -> str:
    m = _NEXT_Q_RE.search(text or "")
    return text[: m.start()] if m else text


def grab(text: str, key: str, fewshot: bool = False) -> str | None:
    """Last match wins — CoT models often restate the key after working.

    `fewshot=True` first cuts everything from the model's own fabricated
    follow-up question onward. Only the few-shot (base-model) path needs this;
    applying it to chat output could truncate a CoT block that legitimately
    contains a "Q:" line.
    """
    if fewshot:
        text = truncate_fewshot_continuation(text)
    ms = key_regex(key).findall(text or "")
    for v in reversed(ms):
        if v.strip():
            return v.strip()
    return None


def clean_answer(raw: str | None) -> str | None:
    if raw is None:
        return None
    a = raw.strip().strip("`").strip()
    a = re.sub(r"^(the answer is|answer is|it is|it's)\s*", "", a, flags=re.I)
    a = a.split("\n")[0].strip()
    a = a.rstrip(".").strip()
    return a or None


def parse_confidence_numeric(raw: str | None) -> float | None:
    """Format A's stated confidence, as a probability in [0, 1].

    The prompt asks for "an integer from 0 to 100", and models comply most of
    the time. When they do not, the original parser divided by 100 regardless:
    a model answering "0.9" was recorded as **0.009**, a hundredfold error that
    lands inside [0, 1], is a distinct value (so the degeneracy pre-flight
    cannot see it), and reads as near-total uncertainty from a model expressing
    near-total confidence. Survivable while Format B was canonical; not now
    that AUDIT finding 2's repair makes A the canonical verbal axis.

    The scale is therefore inferred:

    * ``50/50``, ``fifty-fifty``       -> 0.5             (idiom, not a ratio)
    * ``7/10``, ``1 in 10``, ``2/3``   -> 0.7, 0.1, 0.667  (small-denominator ratio)
    * ``85%``                          -> 0.85            (explicit percentage)
    * ``0.9``, ``.85``, ``1.0``        -> 0.9, 0.85, 1.0  (decimal at or below 1)
    * ``85``, ``100``, ``85.5``        -> 0.85, 1.0, 0.855 (the 0-100 scale asked for)

    The ratio branch requires a denominator of at most RATIO_MAX_DENOM (or
    exactly 100, which is a percentage written long). Anything else returns
    None rather than falling through: the fall-through hands the NUMERATOR to
    the 0-100 branch, so ``"21 out of 21"`` would read 0.21 against 1.0 for
    ``"20 out of 20"`` — a 5x cliff either side of the bound, wrong by exactly
    den/100, and plausible rather than absent. An unparsed row shows up in the
    format-compliance rate; 0.21 does not.

    ``"50/50"`` is handled before that branch, because it is an idiom rather
    than a ratio: read literally it is 1.0, i.e. certainty, which is the
    opposite of what the phrase means. Left to the ratio branch it would record
    perfect confidence for a model saying it had none — manufacturing exactly
    the unwarranted stated confidence this project measures.

    Two known limits, both inherent to inferring a scale from free text:

    * A bare integer ``1`` stays 0.01. The instruction was 0-100, and reading
      it as certainty would invent confidence the model never expressed.
    * A ratio whose denominator is neither at most RATIO_MAX_DENOM nor exactly
      100 returns None. The lookbehind on every branch stops a range like
      ``"90-95%"`` from being read as the negative number -95.
    * The rule is discontinuous at 1.0: ``"1.0"`` is read as a probability
      while ``"1.5"`` is read as 1.5 on the 0-100 scale. A model answering on a
      0-10 scale is misread. There is no reading of a bare "1.5" that is right
      under every convention; the format-compliance rate in `t3_parse_and_accuracy`
      is what tells you whether this matters for a given cell.
    """
    if raw is None:
        return None
    text = raw.strip()
    RATIO_MAX_DENOM = 20

    # "50/50" is an idiom, not a ratio. Read literally it is 50 out of 50 = 1.0,
    # i.e. certainty — the exact opposite of what the phrase means. It is the
    # one x/x form that must NOT go through the ratio branch.
    if re.search(r"(?<![\d.])50\s*[/:-]\s*50(?![\d.])", text) or \
       re.search(r"\bfifty[\s-]*fifty\b", text, re.I):
        return 0.5

    m = re.search(r"(?<![\d.])(-?\d+(?:\.\d+)?)\s*(?:/|out\s+of|in)\s*(-?\d+(?:\.\d+)?)",
                  text, re.I)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if 0 < den <= RATIO_MAX_DENOM and 0 <= num <= den:
            return num / den
        if den == 100 and 0 <= num <= 100:
            return num / 100.0             # "85/100" — a percentage written long
        # Anything else is a ratio on a scale this parser cannot read. Falling
        # through would hand the NUMERATOR to the 0-100 branch and return
        # num/100 instead of num/den — wrong by exactly den/100, and plausible
        # rather than absent: "21 out of 21" would score 0.21 where "20 out of
        # 20" scores 1.0, a 5x cliff either side of the bound. An unparsed row
        # is visible in the format-compliance rate; 0.21 is not.
        return None

    m = re.search(r"(?<![\d.])(-?\d{1,3}(?:\.\d+)?)\s*%", text)
    if m:
        v = float(m.group(1))
        return v / 100.0 if 0 <= v <= 100 else None

    m = re.search(r"(?<![\d.])(-?\d{1,3}(?:\.\d+)?|-?\.\d+)", text)
    if not m:
        return None
    tok = m.group(1)
    v = float(tok)
    if v < 0:                       # a negative confidence is not a parse, it is noise
        return None
    if "." in tok and v <= 1.0:
        return v                                   # already a probability
    if v > 100:
        return None
    return v / 100.0


def parse_confidence_bucket(raw: str | None) -> str | None:
    if raw is None:
        return None
    t = re.sub(r"[^A-Z_ ]", "", raw.upper()).strip().replace(" ", "_")
    if t in BUCKETS:
        return t
    for b in BUCKETS:                      # tolerate "Fairly confident."
        if b in t or t in b:
            return b
    loose = {"CERTAIN": "CERTAIN", "CONFIDENT": "FAIRLY_CONFIDENT", "UNSURE": "SOMEWHAT_UNSURE",
             "GUESSING": "MOSTLY_GUESSING", "GUESS": "MOSTLY_GUESSING", "NO_IDEA": "NO_IDEA",
             "NOIDEA": "NO_IDEA", "UNKNOWN": "NO_IDEA"}
    for k, v in loose.items():
        if k in t:
            return v
    return None


def parse_response(variant: str, text: str, fewshot: bool = False) -> dict:
    """Never raises. `parse_ok` is False when the required fields are absent."""
    out: dict[str, Any] = {"raw_len": len(text or ""), "parse_ok": False}
    ans = clean_answer(grab(text, "ANSWER", fewshot))
    out["reasoning"] = grab(text, "REASONING", fewshot)

    if variant == "A":
        conf = parse_confidence_numeric(grab(text, "CONFIDENCE", fewshot))
        out.update(answer=ans, confidence=conf, parse_ok=ans is not None and conf is not None)
    elif variant == "B":
        bucket = parse_confidence_bucket(grab(text, "CONFIDENCE", fewshot))
        out.update(answer=ans, bucket=bucket, parse_ok=ans is not None and bucket is not None)
    elif variant == "C":
        dec_raw = (grab(text, "DECISION", fewshot) or "").upper()
        decision = "PASS" if "PASS" in dec_raw else ("ANSWER" if "ANSWER" in dec_raw else None)
        if decision is None and ans:                    # no DECISION line but an answer given
            decision = "PASS" if (ans or "").upper() in {"NONE", "PASS", "N/A"} else "ANSWER"
        if decision == "PASS":
            ans = None
        out.update(answer=ans, decision=decision,
                   parse_ok=decision == "PASS" or (decision == "ANSWER" and ans is not None))
    else:                                               # FORCED, SAMPLE
        out.update(answer=ans, parse_ok=ans is not None)
    return out


# %%
# ============================================================================
# CELL 8 — graders: deterministic first, local NLI only where unavoidable
# No LLM judge anywhere. Every item records WHICH tier resolved it, so Gate 1
# agreement is computable per grader family (PLAN §7, §16).
# ============================================================================

ARTICLES = {"a", "an", "the"}
PUNCT_RE = re.compile(r"[^\w\s\.\-/]", re.UNICODE)


def normalize_text(s: str, strip_articles: bool = True) -> str:
    s = (s or "").lower().strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
    s = PUNCT_RE.sub(" ", s)
    toks = [t for t in s.split() if not (strip_articles and t in ARTICLES)]
    return " ".join(toks).strip()


def extract_number(s: str) -> float | None:
    if s is None:
        return None
    t = s.replace(",", "").replace("$", "").replace("%", "").strip()
    ms = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", t)
    if not ms:
        return None
    try:
        return float(ms[-1])                     # final number = the answer
    except ValueError:
        return None


def grade_numeric(pred: str, golds: Sequence[str], tol: float) -> bool | None:
    p = extract_number(pred)
    if p is None:
        return None
    for g in golds:
        gv = extract_number(g)
        if gv is None:
            continue
        if abs(p - gv) <= max(tol, tol * abs(gv)):
            return True
    return False


MATH_SUBS = [
    (r"\\left", ""), (r"\\right", ""), (r"\\!", ""), (r"\\,", ""), (r"\\;", ""), (r"\\ ", " "),
    (r"\\dfrac", r"\\frac"), (r"\\tfrac", r"\\frac"), (r"\\cdot", "*"), (r"\\times", "*"),
    (r"\^\{\\circ\}", ""), (r"\^\\circ", ""), (r"\\%", ""), (r"\\\$", ""), (r"\\text\{([^}]*)\}", r"\1"),
    (r"\\mbox\{([^}]*)\}", r"\1"), (r"\\boxed\{(.*)\}", r"\1"), (r"\s+", ""),
]


def normalize_math(s: str) -> str:
    t = (s or "").strip().strip("$").strip()
    for pat, rep in MATH_SUBS:
        t = re.sub(pat, rep, t)
    t = t.replace("dollars", "").replace("$", "")
    if re.fullmatch(r"-?\d+\.0+", t):
        t = t.split(".")[0]
    return t.lower()


def grade_latex(pred: str, golds: Sequence[str]) -> bool | None:
    if pred is None:
        return None
    if HAS_MATH_VERIFY:
        try:
            from math_verify import parse as mv_parse, verify as mv_verify
            p = mv_parse(f"${pred}$")
            for g in golds:
                if mv_verify(mv_parse(f"${g}$"), p):
                    return True
            return False
        except Exception:                                # noqa: BLE001 - fall through
            pass
    np_ = normalize_math(pred)
    if any(np_ == normalize_math(g) for g in golds):
        return True
    try:                                                 # numeric equivalence as a last resort
        import sympy
        pv = sympy.sympify(np_.replace("\\frac", "").replace("{", "(").replace("}", ")"))
        for g in golds:
            gv = sympy.sympify(normalize_math(g).replace("\\frac", "").replace("{", "(").replace("}", ")"))
            if sympy.simplify(pv - gv) == 0:
                return True
    except Exception:                                    # noqa: BLE001
        return False
    return False


def grade_string(pred: str, golds: Sequence[str], strip_articles: bool) -> bool | None:
    if pred is None:
        return None
    p = normalize_text(pred, strip_articles)
    if not p:
        return None
    gs = [normalize_text(g, strip_articles) for g in golds]
    if p in gs:
        return True
    for g in gs:                                         # containment both ways, guarded by length
        if g and len(g) >= 4 and (g in p or p in g) and abs(len(g) - len(p)) <= max(8, len(g) // 2):
            return True
    return False


class NLIGrader:
    """Local entailment fallback for SimpleQA-style short answers (PLAN §4, §7).

    Deterministic (argmax, no sampling), ~1.6 GB, loaded lazily and only if the
    string tier leaves items unresolved.
    """

    def __init__(self, model_name: str, threshold: float, batch_size: int, dtype, device: str):
        self.model_name, self.threshold, self.batch_size = model_name, threshold, batch_size
        self.dtype, self.device = torch.float32, device  # see _ensure(): fp32 only
        self._tok = None
        self._model = None
        self._entail_idx = 2

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        # Always float32: DeBERTa's disentangled attention has fp32-only kernels
        # and raises "expected scalar type Float but found BFloat16" under the
        # generation dtype. The model is ~1.6 GB, so fp32 costs nothing here.
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        ).to(self.device).eval()
        labels = {v.lower(): k for k, v in self._model.config.id2label.items()}
        self._entail_idx = labels.get("entailment", 2)
        LOG.log("nli_loaded", model=self.model_name, device=self.device)

    @torch.no_grad()
    def entails(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """P(premise entails hypothesis) for each pair."""
        self._ensure()
        scores: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i : i + self.batch_size]
            enc = self._tok([p for p, _ in chunk], [h for _, h in chunk],
                            return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            logits = self._model(**enc).logits.float()
            scores.extend(torch.softmax(logits, -1)[:, self._entail_idx].tolist())
        return scores

    def grade(self, pred: str, golds: Sequence[str], question: str = "") -> bool:
        pairs = [(f"{question} {g}".strip(), f"{question} {pred}".strip()) for g in golds]
        pairs += [(f"{question} {pred}".strip(), f"{question} {g}".strip()) for g in golds]
        s = self.entails(pairs)
        n = len(golds)
        return any(min(s[i], s[i + n]) >= self.threshold for i in range(n))   # bidirectional

    def free(self) -> None:
        self._model, self._tok = None, None
        free_cuda()


def grade_answer(pred: str | None, golds: Sequence[str], answer_form: str, cfg: Config,
                 nli: "NLIGrader | None" = None, question: str = "") -> dict:
    """Tiered grading. Returns {correct, grader, resolved}.

    Tier order: exact/alias -> numeric -> symbolic -> NLI. `resolved=False`
    means no tier could decide (counted, never silently scored as wrong).
    """
    if pred is None or not str(pred).strip():
        return {"correct": False, "grader": "no_answer", "resolved": True}

    if answer_form == "numeric":
        r = grade_numeric(pred, golds, cfg.NUMERIC_TOLERANCE)
        if r is not None:
            return {"correct": r, "grader": "numeric", "resolved": True}
    if answer_form == "latex":
        r = grade_latex(pred, golds)
        if r is not None:
            return {"correct": r, "grader": "symbolic" if HAS_MATH_VERIFY else "latex_normalized", "resolved": True}

    r = grade_string(pred, golds, cfg.STRIP_ARTICLES)
    if r is True:
        return {"correct": True, "grader": "alias_exact", "resolved": True}

    if answer_form in ("entity",) and r is False:
        # PopQA ships an exhaustive alias list — a miss here is a real miss.
        return {"correct": False, "grader": "alias_exact", "resolved": True}

    if cfg.USE_NLI_FALLBACK and nli is not None and answer_form in ("short", "entity"):
        try:
            return {"correct": bool(nli.grade(pred, golds, question)), "grader": "nli", "resolved": True}
        except Exception as exc:                          # noqa: BLE001
            LOG.log("nli_error", error=str(exc)[:120])
    return {"correct": bool(r) if r is not None else False, "grader": "string_fallback", "resolved": r is not None}
# %%
# ============================================================================
# CELL 8b — pre-flight compute + storage estimate (X2 ledger, PLAN §10)
# Run this BEFORE committing a session. The measured ledger at the end
# supersedes it — this is only for deciding N_PER_CELL and the model subset.
# ============================================================================

# Aggregate decode throughput at large batch, HF transformers (not vLLM).
# Bandwidth-bound: tok/s ~= HBM bandwidth / (2 bytes x params), derated.
HW_PROFILES = {
    "blackwell_96gb": dict(bandwidth_tb_s=1.79, efficiency=12.0, label="RTX Pro 6000 Blackwell (molab)"),
    "t4_single":      dict(bandwidth_tb_s=0.32, efficiency=3.0,  label="1x T4 (Kaggle)"),
    "t4_sharded":     dict(bandwidth_tb_s=0.32, efficiency=1.5,  label="2x T4 pipeline-parallel"),
}


def detect_profile() -> str:
    if not DEVICES["cuda"]:
        return "t4_single"
    if DEVICES["max_vram_gb"] > 60:
        return "blackwell_96gb"
    return "t4_sharded" if DEVICES["n_gpu"] > 1 else "t4_single"


def estimate_compute(cfg: Config, profile: str | None = None) -> pd.DataFrame:
    prof = HW_PROFILES[profile or detect_profile()]
    # generations per question: 3 formats + forced pass on ~20% of C-passes + N samples + 1 greedy extraction
    gens = 3 + 0.2 + cfg.N_SAMPLES + 1
    out_tokens_per_model = sum(
        gens * TIER_SPECS[t]["max_new_tokens"] * cfg.N_PER_CELL for t in cfg.active_tiers())
    rows = []
    for m in cfg.active_models():
        spec = MODEL_SPECS[m]
        tps = (prof["bandwidth_tb_s"] * 1e12) / (2 * spec["params_b"] * 1e9) * prof["efficiency"]
        fits = DEVICES["max_vram_gb"] == 0 or spec["params_b"] * 2.1 < DEVICES["max_vram_gb"] - 0.8
        rows.append(dict(
            model=m, params_b=spec["params_b"],
            weights_gb=round(spec["params_b"] * 2, 1),
            fits_one_gpu=bool(fits),
            est_tok_per_s=int(tps),
            est_gpu_hours=round(out_tokens_per_model / tps / 3600, 2),
            activations_mb=round(len(cfg.PERCENTILES) * spec["hidden"] * 4
                                 * cfg.N_PER_CELL * len(cfg.active_tiers()) / 1e6, 1),
        ))
    df = pd.DataFrame(rows)
    total_h = df["est_gpu_hours"].sum()
    print(f"hardware profile : {prof['label']}")
    print(f"N_PER_CELL={cfg.N_PER_CELL}  tiers={len(cfg.active_tiers())}  "
          f"gens/question={gens:.1f}  output tokens/model={out_tokens_per_model/1e6:.1f}M")
    print(df.to_string(index=False))
    print(f"\nestimated total   : {total_h:.2f} GPU-hours "
          f"({total_h / 12:.1f} x 12-hour sessions)")
    print(f"activation storage: {df['activations_mb'].sum()/1000:.2f} GB")
    print(f"weight downloads  : {df['weights_gb'].sum():.1f} GB "
          f"(set PURGE_WEIGHTS=\"after_run\" if storage is tight)")
    if not df["fits_one_gpu"].all():
        print("\n  ! some models exceed a single GPU — device_map='auto' will shard them,")
        print("    which on 2x T4 means pipeline-parallel: roughly half the compute idles.")
    if total_h > 12:
        print("\n  ! exceeds one 12-hour session. Split with ONLY_MODELS across runs;")
        print("    the question bank and checkpoints are shared, so session 2 resumes cleanly.")
    df.to_csv(PATHS["tables"] / "t0_compute_estimate.csv", index=False)
    return df


COMPUTE_ESTIMATE = estimate_compute(CFG)
# %%
# ============================================================================
# CELL 9 — model manager: load / free / concurrency policy
# "after every model run clear gpu memory and checkpoint" is enforced here.
# ============================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def auto_batch_size(model_spec: dict, tier_spec: dict, cfg: Config, n_return: int = 1) -> int:
    if cfg.BATCH_SIZE > 0:
        return cfg.BATCH_SIZE
    if not DEVICES["cuda"]:
        return 4
    free_gb = DEVICES["max_vram_gb"] - model_spec["params_b"] * 2.2
    if free_gb <= 1:
        return 1
    # KV cache scales with layers x hidden x sequence; this is a deliberately
    # conservative heuristic that the runner will halve on OOM anyway.
    # `num_return_sequences` multiplies the resident sequence count, so the
    # N=10 sampling stage costs 10x a single-return stage at the same batch —
    # without this divisor it OOMs on its first batch every time and only
    # recovers via backoff.
    seq = 512 + tier_spec["max_new_tokens"]
    per_seq_gb = 2 * model_spec["layers"] * model_spec["hidden"] * seq * 2 / 1e9 * 1.6
    per_item_gb = per_seq_gb * max(1, n_return)
    bs = int(max(1, min(cfg.BATCH_SIZE_CAP, (free_gb * 0.55) / max(per_item_gb, 1e-6))))
    return max(1, bs)


def device_map(cfg: Config) -> str | dict:
    """One device when the model fits (molab); shard only when it must (2x T4)."""
    if not DEVICES["cuda"]:
        return "cpu"
    return "auto"


class LoadedModel:
    def __init__(self, name: str, model, tokenizer, spec: dict):
        self.name, self.model, self.tokenizer, self.spec = name, model, tokenizer, spec
        self.hidden_device = next(model.parameters()).device


def load_model(name: str, cfg: Config) -> LoadedModel:
    spec = MODEL_SPECS[name]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], padding_side="left", trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"],
        torch_dtype=cfg.resolved_dtype(),
        attn_implementation=cfg.ATTN_IMPL,
        device_map=device_map(cfg),
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    LOG.log("model_loaded", model=name, secs=round(time.time() - t0, 1),
            dtype=str(cfg.resolved_dtype()), vram=vram_report())
    return LoadedModel(name, model, tok, spec)


def free_model(lm: "LoadedModel | None", cfg: Config, purge: bool = False) -> None:
    """The explicit teardown: drop refs, collect, empty the CUDA caching
    allocator, and optionally delete the on-disk snapshot to protect a small
    persistent volume."""
    if lm is None:
        return
    name, hf_id = lm.name, lm.spec["hf_id"]
    try:
        lm.model.to("meta")
    except Exception:                                     # noqa: BLE001
        pass
    lm.model = None
    lm.tokenizer = None
    del lm
    free_cuda()
    if purge and PATHS["hf_cache"]:
        snap = PATHS["hf_cache"] / "hub" / ("models--" + hf_id.replace("/", "--"))
        if snap.exists():
            shutil.rmtree(snap, ignore_errors=True)
            LOG.log("weights_purged", model=name, path=str(snap))
    LOG.log("model_freed", model=name, vram=vram_report())


def plan_model_batches(models: list[str], cfg: Config) -> list[list[str]]:
    """Group models into execution waves per MODEL_EXEC.

    Why the size guard: batched decode is memory-bandwidth bound — every step
    streams the full weight matrix once. Two co-resident 7Bs therefore stream
    2x the bytes for the same token count, so they split throughput rather than
    adding it, while also halving the VRAM available for KV cache. Small models
    are the opposite case: a 0.5B never saturates a 96 GB card, so co-running
    several of them reclaims genuinely idle SMs. Hence: models under
    CONCURRENT_MAX_PARAMS_B may share a wave; larger ones always run alone.

    The same argument is why MODEL_REPLICAS defaults to 1. If you want more
    throughput from one big model, raise BATCH_SIZE — amortising one weight
    read over more sequences beats duplicating the weights.
    """
    if cfg.MODEL_EXEC in ("sequential", "resident"):
        return [[m] for m in models]
    waves, cur = [], []
    for m in models:
        if MODEL_SPECS[m]["params_b"] > cfg.CONCURRENT_MAX_PARAMS_B:
            if cur:
                waves.append(cur)
                cur = []
            waves.append([m])
            continue
        cur.append(m)
        if len(cur) >= cfg.MAX_CONCURRENT_MODELS:
            waves.append(cur)
            cur = []
    if cur:
        waves.append(cur)
    return waves


# %%
# ============================================================================
# CELL 10 — activation extraction via forward hooks (PLAN §6)
# Five percentile layers, last prompt token, single greedy pass, float32.
# `output_hidden_states=True` is deliberately NOT used (blows the output cap).
# ============================================================================


def percentile_layers(n_layers: int, percentiles: Sequence[int]) -> dict[int, int]:
    """percentile -> block index. 0 => embedding output, n_layers => final block."""
    return {p: int(round(p / 100.0 * n_layers)) for p in percentiles}


def _transformer_base(model: Any) -> Any:
    """The decoder stack itself — the module that owns `embed_tokens` and
    `layers` — with no LM head attached.

    PLAN §6 taps depth percentiles of the *prompt* representation, so the
    prefill pass must run the body alone: the logits head is dead weight here
    and, on a 7B model at batch 32, materialising a [32, seq, 152k] logit
    tensor is the difference between fitting in VRAM and not.

    Raises rather than guessing. A silent fallback to `.model` on an
    architecture that nests its stack elsewhere would tap whatever module
    happened to answer to `.layers`, and every downstream AUROC would be
    measuring the wrong thing without a single warning — which is precisely
    the failure this rewrite exists to correct.
    """
    seen = []
    for attr in ("model", "transformer", "gpt_neox", "base_model", "decoder"):
        cand = getattr(model, attr, None)
        if cand is None:
            continue
        seen.append(attr)
        if hasattr(cand, "layers") and hasattr(cand, "embed_tokens"):
            return cand
        inner = getattr(cand, "model", None)          # PEFT / wrapper double-nesting
        if inner is not None and hasattr(inner, "layers") and hasattr(inner, "embed_tokens"):
            return inner
    raise RuntimeError(
        f"ActivationTap: cannot locate the transformer body on "
        f"{type(model).__name__} (checked {seen or 'no known attributes'}). "
        f"The tap requires a module exposing both `embed_tokens` and `layers`. "
        f"Refusing to tap blind — add this architecture to `_transformer_base`."
    )


def last_prompt_index(attention_mask: "torch.Tensor") -> "torch.Tensor":
    """Row-wise index of the final real prompt token, for either padding side.

    `cumsum(-1).argmax(-1)` lands on the last non-pad position regardless of
    where the padding sits, because argmax returns the FIRST occurrence of the
    maximum: left padding [0,0,1,1,1] -> cumsum [0,0,1,2,3] -> 4, right padding
    [1,1,1,0,0] -> cumsum [1,2,3,3,3] -> 2. The old hook hardcoded `-1`, which
    is only correct under left padding and silently taps a PAD embedding the
    moment a tokenizer defaults the other way.
    """
    return attention_mask.long().cumsum(-1).argmax(-1)


class ActivationTap:
    """Captures the hidden vector at the last PROMPT token, at each requested
    depth percentile, during a dedicated prefill forward pass.

    The tap freezes itself once prefill completes. Decoding re-invokes every
    hooked module once per generated token, so an unfrozen buffer would end the
    pass holding the LAST GENERATED token's state — the defect that made a
    provably constant position score AUROC 0.717 (AUDIT finding 1). Freezing is
    what makes the captured vector pre-generation by construction rather than
    by call-ordering luck.
    """

    def __init__(self, lm: LoadedModel, percentiles: Sequence[int]):
        self.lm = lm
        self.map = percentile_layers(lm.spec["layers"], percentiles)
        self.buffer: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._index: "torch.Tensor | None" = None
        self.frozen: bool = True                   # nothing is captured until prefill opens it
        base = _transformer_base(lm.model)
        n_blocks = len(base.layers)
        # Both directions. An OVER-count raises on the index; an UNDER-count
        # would not, and would silently shift every percentile — p100 would tap
        # layers[23] of 28 and still look like "the final block". The depth
        # percentiles are the entire point of Signal 3, so a stale
        # MODEL_SPECS['layers'] must be an error, not a quiet re-scaling.
        # Compare the WHOLE map, not just p100: with PERCENTILES=(0,25,50,75)
        # there is no p100 entry to check, and an under-counted spec would
        # still shift every depth silently.
        truth = percentile_layers(n_blocks, percentiles)
        if self.map != truth:
            raise RuntimeError(
                f"ActivationTap: MODEL_SPECS['layers'] = {lm.spec['layers']} for {lm.name}, "
                f"but the loaded model has {n_blocks} blocks, so every depth percentile is "
                f"mis-mapped ({self.map} vs {truth}). Fix MODEL_SPECS before running."
            )
        try:
            for pct, idx in self.map.items():
                if not 0 <= idx <= n_blocks:
                    raise RuntimeError(
                        f"ActivationTap: percentile {pct} maps to block {idx}, outside "
                        f"[0, {n_blocks}] for {lm.name}."
                    )
                module = base.embed_tokens if idx == 0 else base.layers[idx - 1]
                self._handles.append(module.register_forward_hook(self._make_hook(pct)))
        except Exception:
            # Hooks registered before the failure would otherwise stay attached
            # to the model for the life of the process, with no handle left to
            # remove them.
            self.close()
            raise

    def _make_hook(self, pct: int) -> Callable:
        def hook(_module, _inp, out):
            if self.frozen:
                return
            h = out[0] if isinstance(out, tuple) else out
            if self._index is None:
                sel = h[:, -1, :]
            else:
                idx = self._index.to(h.device)
                sel = h[torch.arange(h.shape[0], device=h.device), idx, :]
            self.buffer[pct] = sel.detach().to(torch.float32).cpu()
        return hook

    def arm(self, attention_mask: "torch.Tensor") -> None:
        """Open the tap for one prefill pass over this batch."""
        self._index = last_prompt_index(attention_mask)
        self.buffer = {}
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True

    def pop(self) -> dict[int, np.ndarray]:
        out = {p: t.numpy() for p, t in self.buffer.items()}
        self.buffer = {}
        return out

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@torch.no_grad()
def prefill_capture(lm: LoadedModel, enc: dict, tap: ActivationTap) -> dict[int, np.ndarray]:
    """One forward pass over the prompt through the transformer body, with the
    tap armed. Returns last-prompt-token activations and leaves the tap FROZEN,
    so the `generate()` call that follows cannot overwrite them."""
    body = _transformer_base(lm.model)
    tap.arm(enc["attention_mask"])
    try:
        body(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
    finally:
        tap.freeze()
    acts = tap.pop()
    missing = sorted(set(tap.map) - set(acts))
    if missing:
        raise RuntimeError(
            f"ActivationTap: prefill produced no activation at percentiles {missing} "
            f"for {lm.name}. Hooks did not fire — the tapped modules are not on the "
            f"forward path of {type(body).__name__}."
        )
    return acts


def finiteness_stats(arr: np.ndarray) -> dict:
    """Gate 3 pre-check (PLAN §16): dirty activations must be distinguishable
    from genuine absence of signal."""
    n = arr.size
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    finite = arr[np.isfinite(arr)]
    return {
        "n": n,
        "nonfinite_frac": (n_nan + n_inf) / n if n else 0.0,
        "nan_frac": n_nan / n if n else 0.0,
        "inf_frac": n_inf / n if n else 0.0,
        "absmax": float(np.abs(finite).max()) if finite.size else float("nan"),
        "std": float(finite.std()) if finite.size else float("nan"),
    }


# %%
# ============================================================================
# CELL 11 — batched generation engine (resumable, OOM-adaptive)
# ============================================================================


def _stop_token_ids(tok, model) -> set[int]:
    """Every id that terminates a sequence, so trailing padding is excluded
    from both the log-probability sum and the length it is divided by."""
    ids: set[int] = set()
    for v in (getattr(tok, "pad_token_id", None), getattr(tok, "eos_token_id", None),
              getattr(getattr(model, "generation_config", None), "eos_token_id", None),
              getattr(getattr(model, "generation_config", None), "pad_token_id", None)):
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple, set)):
            ids.update(int(x) for x in v if isinstance(x, int))
    return ids


@torch.no_grad()
def sequence_logprobs(lm: LoadedModel, sequences: "torch.Tensor", prompt_len: int,
                      stop_ids: "set[int]", attention_mask: "torch.Tensor | None" = None,
                      row_chunk: int = 4, tok_chunk: int = 64) -> list[float]:
    """Mean per-token log-probability of each generated continuation, under the
    model's OWN distribution.

    PLAN §5 weights semantic clusters by probability mass rather than sample
    counts — the difference between Kuhn-style semantic entropy and a histogram
    of ten strings (AUDIT finding 3: zero log-probabilities were computed
    anywhere).

    Why a separate teacher-forced pass, rather than reading them off generation:

    * `output_scores=True` makes `generate()` retain one
      `[batch x n_return, vocab]` float tensor **per generated token**. At
      Qwen2.5's ~152k vocabulary a 512-new-token N=10 sampling batch is tens of
      gigabytes — more than the KV cache `auto_batch_size` budgets for, and on
      some cells more than the card holds. CELL 10 already refuses
      `output_hidden_states=True` for exactly this reason.
    * A `LogitsProcessor` avoids that cost but does not see a well-defined
      quantity: HF applies custom processors *before* the sampling warpers, so
      it reads logits with `repetition_penalty` applied but not temperature or
      top-p — neither the model likelihood nor the sampler's distribution, and
      the ordering is an implementation detail that can change between
      versions. Measured directly: it differs from both.

    So the likelihood is recomputed exactly. This is the raw model
    distribution, unwarped by SAMPLE_TEMPERATURE / SAMPLE_TOP_P, which is the
    quantity the semantic-entropy literature uses.

    Length-normalised (mean, not sum): a long correct answer must not rank
    below a short wrong one purely for being long.

    Cost, stated plainly: this pass sees `prompt_len + max_new` token positions
    per row, against `max_new` for the whole decode loop — roughly 2.7x (C3) to
    6x (R1) the positions, i.e. it about doubles the ARITHMETIC of the SAMPLE
    stage.

    In wall clock that is ~1.35x at batch size 1, where decode is
    memory-bandwidth bound and this pass is not — but it approaches the full
    2x at the batch sizes `auto_batch_size` actually picks (7-8 for the 7B
    models), because at those sizes decode has crossed into compute-bound too
    and the bandwidth discount disappears. It is the largest single cost added
    by the log-probability repair, and `ENTROPY_LP_WEIGHTED=False` turns it
    off. The measured `compute_ledger` captures it; the a-priori budget
    estimate does not.

    Memory is bounded by asking for only the tail of the logits
    (`logits_to_keep`) and by chunking over rows and over time. `tok_chunk`
    bounds only the float32 copy; the `[row_chunk, kept_positions, vocab]`
    tensor the forward pass returns is what `row_chunk` bounds.

    `attention_mask` is REQUIRED whenever prompts were padded, which is always:
    the tokenizer pads left, and `generate()`'s output keeps that padding. A
    plain forward without the mask lets every row attend to its own pad tokens
    AND numbers positions from 0 instead of from the first real token, so RoPE
    shifts. Measured on a mixed-length batch, the padded row's score moved by
    1e-2 while the unpadded row was exact — small enough to look like noise and
    wrong on exactly the rows with the shortest prompts.
    """
    n_rows = int(sequences.shape[0])
    total_len = int(sequences.shape[1])
    if total_len <= prompt_len:
        return [float("nan")] * n_rows
    if attention_mask is None:
        attention_mask = torch.ones_like(sequences)
    elif attention_mask.shape != sequences.shape:
        raise ValueError(
            f"sequence_logprobs: attention_mask {tuple(attention_mask.shape)} does not match "
            f"sequences {tuple(sequences.shape)} — it must cover prompt AND continuation.")
    # Same position numbering generate() uses under left padding.
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    out: list[float] = []
    base = lm.model
    rc = max(1, int(row_chunk))
    r0 = 0
    while r0 < n_rows:
        sl = sequences[r0 : r0 + rc]
        am = attention_mask[r0 : r0 + rc]
        pos = position_ids[r0 : r0 + rc]
        want_positions = int(sl.shape[1]) - prompt_len + 1     # continuation + its predictor
        try:
            try:
                # Ask for only the tail of the logits. The prompt-position
                # logits are sliced away immediately anyway, and they are most
                # of the tensor: at C3 sizes this is the difference between
                # 1.48 GB and 0.55 GB per chunk.
                logits = base(input_ids=sl, attention_mask=am, position_ids=pos,
                              use_cache=False, logits_to_keep=want_positions).logits
                trimmed = int(logits.shape[1]) == want_positions
            except TypeError:
                logits = base(input_ids=sl, attention_mask=am, position_ids=pos,
                              use_cache=False).logits
                trimmed = False
        except torch.cuda.OutOfMemoryError:
            # A [rows, seq, 152k] logits tensor is the whole reason this is
            # chunked; back off the same way the generation loop does.
            free_cuda()
            if rc == 1:
                LOG.log("logprob_skip", rows=int(sl.shape[0]),
                        note="OOM at row_chunk=1 — these samples fall back to count weighting")
                out += [float("nan")] * int(sl.shape[0])
                r0 += 1
                continue
            rc = max(1, rc // 2)
            LOG.log("logprob_backoff", new_row_chunk=rc)
            continue
        tgt = sl[:, prompt_len:]                                   # [b, T]
        # position t of the continuation was predicted from index t-1
        pred = (logits[:, :-1, :] if trimmed
                else logits[:, prompt_len - 1 : total_len - 1, :])  # [b, T, V]
        b, T = tgt.shape
        got = torch.empty((b, T), dtype=torch.float32, device=tgt.device)
        for t0 in range(0, T, max(1, tok_chunk)):
            sub = pred[:, t0 : t0 + max(1, tok_chunk), :].to(torch.float32)
            ids = tgt[:, t0 : t0 + max(1, tok_chunk)]
            chosen = sub.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
            got[:, t0 : t0 + ids.shape[1]] = chosen - torch.logsumexp(sub, dim=-1)
            del sub
        del logits, pred
        # Everything at and after the first stop token is padding.
        stop = torch.zeros_like(tgt, dtype=torch.bool)
        for sid in stop_ids:
            stop |= (tgt == sid)
        alive = (~stop).to(torch.int32).cumprod(dim=1).bool()
        cnt = alive.sum(dim=1)
        tot = (got * alive).sum(dim=1)
        out += [float(t / c) if int(c) > 0 else float("nan") for t, c in zip(tot, cnt)]
        r0 += int(sl.shape[0])
    return out


@torch.no_grad()
def generate_batch(lm: LoadedModel, prompts: list[str], max_new_tokens: int,
                   temperature: float, top_p: float, n_return: int,
                   tap: "ActivationTap | None" = None
                   ) -> tuple[list[list[str]], dict, "list[list[float]] | None"]:
    tok = lm.tokenizer
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536)
    enc = {k: v.to(lm.hidden_device) for k, v in enc.items()}

    # PLAN §6 / CHANGESforPLANv3 §4: the internal signal is a PRE-GENERATION
    # reading of the prompt. Capture it in its own forward pass BEFORE decode
    # starts, then freeze the tap. Reading the hook buffer after `generate()`
    # — as this did until the audit — returns the last GENERATED token instead,
    # which is a during-generation probe of a different quantity entirely.
    acts = prefill_capture(lm, enc, tap) if tap is not None else {}

    do_sample = temperature and temperature > 0
    # Sequence log-probs only matter for the multi-sample entropy pass. They
    # come from a separate teacher-forced pass rather than `output_scores=True`
    # — see `sequence_logprobs` for why that flag would cost tens of gigabytes
    # on this grid, and why a logits processor does not measure a well-defined
    # quantity either.
    want_lp = bool(n_return > 1)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=bool(do_sample),
        num_return_sequences=n_return,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=False,
    )
    if do_sample:
        gen_kwargs.update(temperature=float(temperature), top_p=float(top_p))
    out = lm.model.generate(**enc, **gen_kwargs)
    plen = enc["input_ids"].shape[1]
    if want_lp:
        # generate() expands each prompt into n_return rows in order, so the
        # prompt mask repeats the same way. The continuation is all-real for
        # scoring purposes: causal attention means a trailing pad can only
        # affect positions after it, and those are dropped by the stop mask.
        pm = enc["attention_mask"].repeat_interleave(n_return, dim=0) if n_return > 1 \
            else enc["attention_mask"]
        full_mask = torch.cat(
            [pm, torch.ones((out.shape[0], out.shape[1] - plen), dtype=pm.dtype, device=pm.device)],
            dim=1)
        logprobs = sequence_logprobs(lm, out, plen, _stop_token_ids(tok, lm.model),
                                     attention_mask=full_mask)
    else:
        logprobs = None
    texts = tok.batch_decode(out[:, plen:], skip_special_tokens=True)
    grouped = [texts[i * n_return : (i + 1) * n_return] for i in range(len(prompts))]
    lp_grouped = ([logprobs[i * n_return : (i + 1) * n_return] for i in range(len(prompts))]
                  if logprobs is not None else None)
    return grouped, acts, lp_grouped


def run_generation(lm: LoadedModel, items: list[dict], variant: str, tier: str, cfg: Config,
                   ckpt: Checkpoint, n_return: int = 1, temperature: float | None = None,
                   capture_activations: bool = False) -> dict:
    """Drive one (model, tier, variant) pass with resume, OOM backoff, and
    periodic cache clearing. Returns a stats dict for the ledger."""
    tier_spec = TIER_SPECS[tier]
    todo = [it for it in items if not ckpt.has(qid=it["qid"], variant=variant)]
    stats = {"requested": len(items), "todo": len(todo), "generated": 0, "parse_ok": 0,
             "oom_backoffs": 0, "seconds": 0.0, "out_tokens": 0}
    if not todo:
        return stats

    bs = auto_batch_size(lm.spec, tier_spec, cfg, n_return)
    temp = cfg.GREEDY_TEMPERATURE if temperature is None else temperature
    tap = ActivationTap(lm, cfg.PERCENTILES) if capture_activations else None
    act_store: dict[int, list[np.ndarray]] = defaultdict(list)
    act_qids: list[str] = []
    t0 = time.time()
    n_batches = 0

    try:
        pbar = tqdm(total=len(todo), desc=f"{lm.name[:18]}|{tier}|{variant}", leave=False)
        i = 0
        while i < len(todo):
            chunk = todo[i : i + bs]
            prompts = [build_prompt(variant, it["question"], tier_spec, lm.spec, lm.tokenizer, cfg)
                       for it in chunk]
            try:
                gens, acts, lps = generate_batch(lm, prompts, tier_spec["max_new_tokens"], temp,
                                                 cfg.SAMPLE_TOP_P, n_return, tap)
            except torch.cuda.OutOfMemoryError:
                free_cuda()
                stats["oom_backoffs"] += 1
                if bs == 1:
                    LOG.log("oom_skip", model=lm.name, tier=tier, variant=variant, qid=chunk[0]["qid"])
                    i += 1
                    pbar.update(1)
                    continue
                bs = max(1, bs // 2)
                LOG.log("oom_backoff", model=lm.name, tier=tier, variant=variant, new_batch=bs)
                continue

            for j, it in enumerate(chunk):
                texts = gens[j]
                # base models have no chat template and run few-shot, so their
                # output must be cut at the first self-generated question
                parsed = [parse_response(variant, t, fewshot=not lm.spec["chat"])
                          for t in texts]
                rec = {
                    "qid": it["qid"], "variant": variant, "tier": tier, "model": lm.name,
                    "split": it["split"], "is_pilot": it.get("is_pilot", False),
                    "is_agreement": it.get("is_agreement", False),
                    "n_return": n_return, "temperature": temp,
                    "parsed": parsed if n_return > 1 else parsed[0],
                    "parse_ok": (sum(p["parse_ok"] for p in parsed) / len(parsed)) if n_return > 1
                                else parsed[0]["parse_ok"],
                    "config_hash": cfg.hash(), "code_sha": PROV["code_sha"], "seed": cfg.SEED,
                }
                if lps is not None:
                    rec["seq_logprob"] = [None if not np.isfinite(x) else float(x) for x in lps[j]]
                if cfg.SAVE_RAW_TEXT:
                    rec["raw"] = texts if n_return > 1 else texts[0]
                ckpt.add(rec)
                stats["generated"] += 1
                stats["parse_ok"] += float(rec["parse_ok"])
                stats["out_tokens"] += sum(len(t) for t in texts) // 4

            if acts:
                for p, arr in acts.items():
                    act_store[p].append(arr)
                act_qids.extend(it["qid"] for it in chunk)

            i += len(chunk)
            n_batches += 1
            pbar.update(len(chunk))
            if cfg.EMPTY_CACHE_EVERY_BATCHES and n_batches % cfg.EMPTY_CACHE_EVERY_BATCHES == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        pbar.close()
    finally:
        ckpt.flush()
        if tap is not None:
            tap.close()

    if act_store and cfg.SAVE_ACTIVATIONS:
        save_activations(lm.name, tier, act_qids, act_store, cfg)

    stats["seconds"] = round(time.time() - t0, 1)
    stats["parse_rate"] = stats["parse_ok"] / max(stats["generated"], 1)
    return stats


def save_activations(model: str, tier: str, qids: list[str],
                      store: dict[int, list[np.ndarray]], cfg: Config) -> None:
    """Append-safe shard per (model, tier). Stored float32 per PLAN §16."""
    path = PATHS["acts"] / f"{model}__{tier}.npz"
    payload = {"qids": np.array(qids, dtype=object)}
    fin: dict[str, dict] = {}
    for p, chunks in store.items():
        arr = np.concatenate(chunks, axis=0).astype(cfg.PROBE_STORE_DTYPE)
        payload[f"p{p}"] = arr
        fin[f"p{p}"] = finiteness_stats(arr)
    if path.exists():                                # merge with a previous session
        old = np.load(path, allow_pickle=True)
        merged = {"qids": np.concatenate([old["qids"], payload["qids"]])}
        for p in cfg.PERCENTILES:
            k = f"p{p}"
            if k in old and k in payload:
                merged[k] = np.concatenate([old[k], payload[k]], axis=0)
        # De-duplicate by qid, keeping the most recent row. The checkpoint
        # normally prevents re-extracting a question at all, but it keys on
        # (qid, variant) — so renaming the variant makes every question look
        # new and would append a second copy of the whole shard. Duplicated
        # rows would silently double the probe's training set and break the
        # independence its AUROC assumes.
        last = {}
        for i, q in enumerate(merged["qids"]):
            last[str(q)] = i
        keep = np.array(sorted(last.values()), dtype=int)
        if len(keep) != len(merged["qids"]):
            LOG.log("activations_deduped", model=model, tier=tier,
                    before=len(merged["qids"]), after=len(keep))
        payload = {k: v[keep] for k, v in merged.items()}
    (np.savez_compressed if cfg.COMPRESS_ACTIVATIONS else np.savez)(path, **payload)
    json_write(PATHS["acts"] / f"{model}__{tier}.finiteness.json", fin)
    LOG.log("activations_saved", model=model, tier=tier, n=len(payload["qids"]),
            nonfinite=max((v["nonfinite_frac"] for v in fin.values()), default=0.0))


def purge_all_weights() -> None:
    """PURGE_WEIGHTS='after_run' — drop the whole hub cache once, at the end."""
    hub = (PATHS["hf_cache"] / "hub") if PATHS["hf_cache"] else None
    if hub and hub.exists():
        n = sum(1 for _ in hub.glob("models--*"))
        shutil.rmtree(hub, ignore_errors=True)
        LOG.log("weights_purged_all", snapshots=n, path=str(hub))
# %%
# ============================================================================
# CELL 12 — per-model stage drivers (generation side)
# ============================================================================

RAW_KEYS = ("qid", "variant")


def raw_path(stage: str, model: str, tier: str) -> Path:
    return PATHS["raw"] / stage / model / f"{tier}.jsonl"


def open_ckpt(stage: str, model: str, tier: str, cfg: Config) -> Checkpoint:
    return Checkpoint(raw_path(stage, model, tier), RAW_KEYS, cfg.CHECKPOINT_EVERY, cfg.RESUME)


def cell_items(bank: dict, tier: str, subset: str) -> list[dict]:
    rows = bank[tier]
    if subset == "pilot":
        return [r for r in rows if r.get("is_pilot")]
    if subset == "agreement":
        return [r for r in rows if r.get("is_agreement")]
    return rows


def stage_pilot(lm: LoadedModel, bank: dict, cfg: Config) -> dict:
    """100 questions/cell, Format FORCED (plain answer), to place the cell in
    the 25–80% accuracy band before committing compute (PLAN §3)."""
    out = {}
    for tier in cfg.active_tiers():
        with open_ckpt("pilot", lm.name, tier, cfg) as ck:
            out[tier] = run_generation(lm, cell_items(bank, tier, "pilot"), "FORCED", tier, cfg, ck)
    return out


def stage_verbal(lm: LoadedModel, bank: dict, cfg: Config, committed: set[str]) -> dict:
    """Signal 1 — formats A/B/C (PLAN §4). Formats run on the agreement subset
    for every cell (H0/Gate 2 needs no commitment) and on the full cell only
    where the cell is committed."""
    out = {}
    for tier in cfg.active_tiers():
        full = cell_id(lm.name, tier) in committed
        items = cell_items(bank, tier, "all" if full else "agreement")
        with open_ckpt("verbal", lm.name, tier, cfg) as ck:
            for variant in ("A", "B", "C"):
                out[f"{tier}:{variant}"] = run_generation(lm, items, variant, tier, cfg, ck)
    return out


def stage_forced(lm: LoadedModel, bank: dict, cfg: Config, committed: set[str]) -> dict:
    """PLAN §4.1 — forced-answer companion on every Format C Pass."""
    out = {}
    by_qid = {t: {r["qid"]: r for r in bank[t]} for t in cfg.active_tiers()}
    for tier in cfg.active_tiers():
        recs = jsonl_read(raw_path("verbal", lm.name, tier))
        passes = [r["qid"] for r in recs
                  if r["variant"] == "C" and isinstance(r.get("parsed"), dict)
                  and r["parsed"].get("decision") == "PASS"]
        items = [by_qid[tier][q] for q in dict.fromkeys(passes) if q in by_qid[tier]]
        if not items:
            out[tier] = {"requested": 0, "todo": 0, "generated": 0, "note": "no Format C passes"}
            continue
        with open_ckpt("forced", lm.name, tier, cfg) as ck:
            out[tier] = run_generation(lm, items, "FORCED", tier, cfg, ck)
    return out


def stage_sample(lm: LoadedModel, bank: dict, cfg: Config, committed: set[str]) -> dict:
    """Signal 2 — N=10 generations at T in [0.7, 1.0] (PLAN §5)."""
    assert cfg.SAMPLE_TEMPERATURE >= 0.7, "PLAN §5: sampling needs real variance (T >= 0.7)"
    out = {}
    for tier in cfg.active_tiers():
        if cell_id(lm.name, tier) not in committed:
            continue
        with open_ckpt("sample", lm.name, tier, cfg) as ck:
            out[tier] = run_generation(lm, cell_items(bank, tier, "all"), "SAMPLE", tier, cfg, ck,
                                       n_return=cfg.N_SAMPLES, temperature=cfg.SAMPLE_TEMPERATURE)
    return out


def assert_prompt_alignment(lm: LoadedModel, bank: dict, cfg: Config,
                             a: str = "SAMPLE", b: str = "EXTRACT") -> None:
    """Two variants that are meant to share a generation context must RENDER
    identically, not merely be written to.

    AUDIT finding 4: each signal is measured on a different generation, and
    accuracy differs by up to 12 points across variants. SAMPLE and EXTRACT
    share an `instruction()` branch and a FEWSHOT_POOL entry today, so the
    behavioral and internal signals read the same context — but nothing
    enforced that, and a one-line edit to either branch would silently
    decouple them. This fails loudly at runtime instead.
    """
    for tier in cfg.active_tiers():
        spec = TIER_SPECS[tier]
        items = cell_items(bank, tier, "all")[:1]
        if not items:
            continue
        q = items[0]["question"]
        pa = build_prompt(a, q, spec, lm.spec, lm.tokenizer, cfg)
        pb = build_prompt(b, q, spec, lm.spec, lm.tokenizer, cfg)
        if pa != pb:
            raise RuntimeError(
                f"prompt drift between {a} and {b} on tier {tier} for {lm.name}. "
                f"These variants must render byte-identically or the behavioral and internal "
                f"signals are measured on different generations (AUDIT finding 4).\n"
                f"--- {a} ---\n{pa!r}\n--- {b} ---\n{pb!r}")
    LOG.log("prompt_alignment_ok", model=lm.name, variants=[a, b],
            tiers=list(cfg.active_tiers()))


def stage_extract(lm: LoadedModel, bank: dict, cfg: Config, committed: set[str]) -> dict:
    """Signal 3 — single greedy pass, five percentile taps at the last prompt
    token (PLAN §6). Uses the SAMPLE prompt so the probe reads a plain
    answering context, not a confidence-elicitation context."""
    assert_prompt_alignment(lm, bank, cfg)
    out = {}
    for tier in cfg.active_tiers():
        if cell_id(lm.name, tier) not in committed:
            continue
        with open_ckpt("extract", lm.name, tier, cfg) as ck:
            out[tier] = run_generation(lm, cell_items(bank, tier, "all"), "EXTRACT", tier, cfg, ck,
                                       temperature=0.0, capture_activations=True)
    return out


def evaluate_band_gate(bank: dict, cfg: Config, nli: "NLIGrader | None") -> dict:
    """Grade the pilot and decide cell commitment (PLAN §3, §10 — a ragged grid
    is the planned outcome, not a failure)."""
    verdicts = {}
    for model in cfg.active_models():
        for tier in cfg.active_tiers():
            recs = jsonl_read(raw_path("pilot", model, tier))
            if not recs:
                continue
            gold = {r["qid"]: r for r in bank[tier]}
            n_ok, n_tot = 0, 0
            for r in recs:
                q = gold.get(r["qid"])
                if q is None:
                    continue
                p = r["parsed"] if isinstance(r["parsed"], dict) else r["parsed"][0]
                g = grade_answer(p.get("answer"), q["answers"], q["answer_form"], cfg, nli, q["question"])
                n_ok += int(g["correct"])
                n_tot += 1
            acc = n_ok / n_tot if n_tot else 0.0
            lo, hi = cfg.ACCURACY_BAND
            in_band = lo <= acc <= hi
            committed = bool(in_band or cfg.COMMIT_CELLS_OUTSIDE_BAND)
            verdicts[cell_id(model, tier)] = {
                "model": model, "tier": tier, "n": n_tot, "accuracy": round(acc, 4),
                "band": [lo, hi], "in_band": in_band, "committed": committed,
                "pilot_accuracy": round(acc, 4),
                "parse_rate": round(float(np.mean([r["parse_ok"] for r in recs])), 4),
            }
    json_write(PATHS["derived"] / "cell_commitments.json", verdicts)
    n_c = sum(v["committed"] for v in verdicts.values())
    n_out = sum(1 for v in verdicts.values() if not v["in_band"])
    LOG.log("band_gate", cells=len(verdicts), committed=n_c, ragged=len(verdicts) - n_c,
            out_of_band_but_committed=(n_out if cfg.COMMIT_CELLS_OUTSIDE_BAND else 0),
            mode=("covariate" if cfg.COMMIT_CELLS_OUTSIDE_BAND else "deletion"))
    if cfg.COMMIT_CELLS_OUTSIDE_BAND and n_out:
        # Conditioning cell inclusion on accuracy is a legitimate base-rate
        # control and is this project's best claim to novelty. DELETING cells
        # is the wrong way to implement it — it removed R2, R3, 0.5B and 5/6 of
        # C3 on an n=100 pilot whose estimates moved by 16 points between runs
        # (AUDIT finding 9). The band is kept as a covariate; the control is
        # applied by difficulty matching and by the GLMM's cell intercepts.
        LOG.log("band_gate_as_covariate", out_of_band=n_out,
                note="cells outside the 25-80% band are COMMITTED and flagged in_band=False; "
                     "base-rate control is applied statistically, not by deletion")
    return verdicts


# %%
# ============================================================================
# CELL 13 — grading stage + semantic entropy (PLAN §5, §7)
# ============================================================================


def stage_grade(bank: dict, cfg: Config, nli: "NLIGrader | None") -> pd.DataFrame:
    """Grade every stored generation. One row per (model, tier, qid, variant,
    sample_idx) with the resolving grader recorded."""
    ck_path = PATHS["derived"] / "graded.jsonl"
    ck = Checkpoint(ck_path, ("model", "tier", "qid", "variant", "sample_idx"),
                    cfg.CHECKPOINT_EVERY, cfg.RESUME)
    for stage in ("pilot", "verbal", "forced", "sample", "extract"):
        for model in cfg.active_models():
            for tier in cfg.active_tiers():
                recs = jsonl_read(raw_path(stage, model, tier))
                if not recs:
                    continue
                gold = {r["qid"]: r for r in bank[tier]}
                for r in tqdm(recs, desc=f"grade {stage}|{model[:14]}|{tier}", leave=False):
                    q = gold.get(r["qid"])
                    if q is None:
                        continue
                    plist = r["parsed"] if isinstance(r["parsed"], list) else [r["parsed"]]
                    for si, p in enumerate(plist):
                        if ck.has(model=model, tier=tier, qid=r["qid"], variant=r["variant"], sample_idx=si):
                            continue
                        g = grade_answer(p.get("answer"), q["answers"], q["answer_form"], cfg, nli, q["question"])
                        ck.add({
                            "stage": stage, "model": model, "tier": tier, "qid": r["qid"],
                            "variant": r["variant"], "sample_idx": si, "split": r["split"],
                            "is_pilot": r.get("is_pilot", False), "is_agreement": r.get("is_agreement", False),
                            "family": q["family"], "answer_form": q["answer_form"],
                            "answer": p.get("answer"), "gold": q["answers"][0],
                            "confidence": p.get("confidence"), "bucket": p.get("bucket"),
                            "decision": p.get("decision"), "parse_ok": p.get("parse_ok"),
                            **g,
                        })
    ck.flush()
    df = pd.DataFrame(jsonl_read(ck_path))
    df.to_parquet(PATHS["derived"] / "graded.parquet", index=False) if len(df) else None
    LOG.log("graded", rows=len(df),
            graders=dict(Counter(df["grader"])) if len(df) else {},
            unresolved=int((~df["resolved"]).sum()) if len(df) else 0)
    return df


def numeric_equal(a: str, b: str, tol: float = 1e-9) -> bool:
    """Both sides reduced to a FINITE real number, if they reduce at all.

    Absolute tolerance, deliberately. A relative tolerance merges answers that
    are genuinely different: at 1e12 a 1e-9 relative bound is +/-1000, so
    1000000000000 and 1000000000500 would land in one semantic cluster, and
    union-find would then chain a whole run of large integers together —
    under-counting entropy and overstating behavioral confidence on exactly the
    MATH L4-5 answers where large integers appear. Two spellings of the same
    number differ by float-rounding noise, which absolute 1e-9 already covers
    (1/3 vs 0.3333333333 differ by ~3e-11).

    Non-finite values are rejected outright: with vb = +/-inf the tolerance
    bound is itself inf, so inf <= inf holds and +inf would merge with -inf.
    """
    def val(s: str) -> "float | None":
        t = normalize_math(s)
        m = re.fullmatch(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", t) or \
            re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", t)
        try:
            if m:
                den = float(m.group(2))
                return float(m.group(1)) / den if den else None
            return float(t)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    va, vb = val(a), val(b)
    if va is None or vb is None:
        return False
    if not (np.isfinite(va) and np.isfinite(vb)):
        return False
    return abs(va - vb) <= tol


def _numeric_equal(a: str, b: str, tol: float = 1e-9) -> bool:
    return numeric_equal(a, b, tol=tol)


def math_equal(a: str, b: str) -> bool:
    """Symbolic/numeric equivalence between two MODEL answers.

    Reuses `grade_latex`, the same oracle the grader uses against gold, so an
    answer pair that would both be marked correct can never land in two
    different semantic clusters. Checked in both directions because
    `grade_latex`'s normalise-and-contain fallbacks are not symmetric.

    AUDIT finding 3: NLI merging was gated on answer_form in {short, entity},
    so all nine committed GSM8K/MATH cells clustered by exact string alone —
    "0.5", "1/2" and "\\frac{1}{2}" counted as three distinct beliefs and
    inflated the entropy of a model that never wavered.
    """
    if not a or not b:
        return False
    try:
        if grade_latex(a, [b]) or grade_latex(b, [a]):
            return True
    except Exception:                                    # noqa: BLE001
        pass
    # `grade_latex`'s no-math_verify fallback cannot evaluate \frac{a}{b}: it
    # strips the macro and hands sympy "(1)(2)", which raises. Without this
    # guard, clustering quality would depend on whether an optional package
    # happened to install — so evaluate both sides numerically as a last resort.
    return numeric_equal(a, b)


def union_find_merge(keys: list[int], should_merge: Callable[[int, int], bool]) -> dict[int, int]:
    """Merge cluster keys pairwise under `should_merge`; returns key -> root."""
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if find(a) != find(b) and should_merge(a, b):
                parent[find(a)] = find(b)
    return {k: find(k) for k in keys}


def _union_find_merge(keys: list[int], should_merge: Callable[[int, int], bool]) -> dict[int, int]:
    return union_find_merge(keys, should_merge)


def cluster_answers(answers: list[str], answer_form: str, cfg: Config,
                     nli: "NLIGrader | None", question: str = "") -> list[int]:
    """Fast path: normalised string identity. Then math equivalence for
    symbolic forms. Then bidirectional NLI entailment (PLAN §5·2)."""
    norm = [normalize_math(a) if answer_form in ("latex", "numeric")
            else normalize_text(a, cfg.STRIP_ARTICLES) for a in answers]
    labels: list[int] = []
    reps: list[str] = []
    for a, n in zip(answers, norm):
        hit = None
        for k, rn in enumerate(reps):
            if n == rn:
                hit = k
                break
        if hit is None:
            reps.append(n)
            hit = len(reps) - 1
        labels.append(hit)

    def relabel(remap: dict[int, int], labs: list[int]) -> list[int]:
        canon = {v: i for i, v in enumerate(sorted(set(remap.values())))}
        return [canon[remap[l]] for l in labs]

    originals: dict[int, str] = {}
    for a, l in zip(answers, labels):
        originals.setdefault(l, a)
    keys = sorted(originals)

    # --- symbolic equivalence (GSM8K / MATH), BEFORE the NLI pass ----------
    if answer_form in ("latex", "numeric") and len(keys) >= 2:
        remap = union_find_merge(keys, lambda a, b: math_equal(originals[a], originals[b]))
        labels = relabel(remap, labels)
        originals = {}
        for a, l in zip(answers, labels):
            originals.setdefault(l, a)
        keys = sorted(originals)

    # --- NLI entailment; now reachable for math forms too -----------------
    nli_forms = ("short", "entity", "latex", "numeric")
    if not (cfg.USE_NLI_FALLBACK and nli is not None and answer_form in nli_forms):
        return labels
    if len(keys) < 2:
        return labels
    # Give the entailment model the same context `NLIGrader.grade` gives it.
    # Without the question, deberta-large-mnli is asked whether the bare string
    # "18" entails the bare string "72" — a judgement it has no basis for. With
    # it, the pair reads as two answers to the same question.
    def framed(x: str) -> str:
        return f"{question} {x}".strip() if question else x

    pairs, meta = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_txt, b_txt = originals[keys[i]], originals[keys[j]]
            pairs += [(framed(a_txt), framed(b_txt)), (framed(b_txt), framed(a_txt))]
            meta.append((keys[i], keys[j]))
    if not pairs:
        return labels
    try:
        s = nli.entails(pairs)
    except Exception:                                   # noqa: BLE001
        return labels
    verdict = {pair: bool(min(s[2 * idx], s[2 * idx + 1]) >= cfg.NLI_ENTAIL_THRESHOLD)
               for idx, pair in enumerate(meta)}
    if answer_form in ("latex", "numeric"):
        # `math_equal` already made every merge the grading oracle would make,
        # so an ADDITIONAL merge here joins two mathematically distinct answers.
        # Sometimes right (different renderings the oracle could not parse),
        # often not — either way it must be visible, not silent.
        extra = [(originals[a], originals[b]) for (a, b), v in verdict.items() if v]
        if extra:
            LOG.log("nli_merged_math_answers", answer_form=answer_form,
                    n_merges=len(extra), pairs=extra[:10],
                    note="these answers are NOT equal under the symbolic oracle; "
                         "set USE_NLI_FALLBACK=False to disable")
    remap = union_find_merge(
        keys, lambda a, b: verdict.get((a, b), False) or verdict.get((b, a), False))
    return relabel(remap, labels)


def cluster_mass(labels: list[int], logprobs: "list[float] | None",
                 cfg: Config) -> tuple[np.ndarray, bool]:
    """Probability mass per semantic cluster.

    Two weightings:

    * **counts** — a cluster's mass is how many samples landed in it. This is
      what the pipeline did before the audit: a histogram of ten strings.
    * **length-normalised likelihood** — a cluster's mass is the summed
      `exp(mean per-token log-probability)` of its members, i.e. the summed
      GEOMETRIC-MEAN per-token probability. This is length normalisation in
      Kuhn's sense, and it is deliberately NOT the raw sequence likelihood:
      un-normalised, a 200-token chain-of-thought is astronomically less
      likely than a 3-token answer purely for being longer, and the long
      cluster would vanish regardless of how confident the model was.

    The consequence of normalising is worth stating plainly: a cluster can hold
    fewer samples than another and still carry more mass. That is the intended
    behaviour — it is what "weight by probability, not by count" means — but it
    is why `modal_share` (a sample fraction) and `modal_mass` (a mass fraction)
    are reported as two different columns rather than one.

    Falls back to counts for checkpoints written before log-probs were
    recorded, and reports which path was taken.
    """
    keys = sorted(set(labels))
    counts = np.array([sum(1 for l in labels if l == k) for k in keys], dtype=float)
    if not cfg.ENTROPY_LP_WEIGHTED or not logprobs:
        return counts, False
    lp = np.array([np.nan if x is None else float(x) for x in logprobs], dtype=float)
    if len(lp) != len(labels) or not np.isfinite(lp).all():
        return counts, False
    shift = float(np.max(lp))                    # shared shift, for numerical range only
    w = np.exp(lp - shift)
    mass = np.array([w[[i for i, l in enumerate(labels) if l == k]].sum() for k in keys],
                    dtype=float)
    if not np.isfinite(mass).all() or mass.sum() <= 0:
        return counts, False
    return mass, True


def stage_entropy(bank: dict, cfg: Config, nli: "NLIGrader | None") -> pd.DataFrame:
    """Shannon entropy over semantic cluster mass -> behavioral confidence."""
    rows = []
    n_req = int(cfg.N_SAMPLES)
    # PLAN §5 asks for confidence relative to N=10 REQUESTED samples. Dividing
    # by log(n_valid) instead let a question with a single parsed sample score
    # a perfect 1.0 — maximum confidence certified by one observation (AUDIT
    # finding 3, 7.9% of rows). The denominator is now fixed across questions,
    # and a question below ENTROPY_MIN_VALID is NaN rather than guessed at.
    H_max_fixed = float(np.log(n_req)) if n_req > 1 else 0.0
    for model in cfg.active_models():
        for tier in cfg.active_tiers():
            recs = jsonl_read(raw_path("sample", model, tier))
            if not recs:
                continue
            gold = {r["qid"]: r for r in bank[tier]}

            # ---- per-cell weighting decision, BEFORE scoring anything ----
            # `confidence_behavioral` is calibrated per cell and median-split
            # per cell, so the whole cell has to be on one scale. A resumed run
            # mixes records: those written before log-probs existed have none,
            # and RESUME skips re-generating them. Counting some questions and
            # weighting others would fit one calibrator over two different
            # quantities, so the cell falls back to counts wholesale.
            def _has_lp(r: dict) -> bool:
                """Does this record have a usable log-prob for every sample that
                will actually be CLUSTERED?

                Requiring all N to be finite is far too strict: a sample whose
                continuation begins with a stop token scores NaN, and that same
                sample produced no parseable answer, so `keep` discards it
                anyway. The OOM path writes NaN for the same reason. At
                N_PER_CELL=1000 one such sample somewhere in the cell is close
                to certain, and it would silently switch the entire cell back to
                count weighting — disabling the Rao-Blackwellisation this repair
                exists to add, with a single log line as the only evidence.
                """
                lp = r.get("seq_logprob")
                parsed = r["parsed"] if isinstance(r["parsed"], list) else [r["parsed"]]
                if not (isinstance(lp, list) and len(lp) == len(parsed)):
                    return False
                kept = [i for i, pp in enumerate(parsed) if pp.get("answer")]
                if not kept:
                    return True            # nothing to weight; not evidence of a problem
                return all(lp[i] is not None and np.isfinite(float(lp[i])) for i in kept)

            n_with_lp = sum(1 for r in recs if _has_lp(r))
            cell_lp_ok = bool(cfg.ENTROPY_LP_WEIGHTED and n_with_lp == len(recs) and len(recs))
            if cfg.ENTROPY_LP_WEIGHTED and 0 < n_with_lp < len(recs):
                LOG.log("entropy_mixed_weighting", model=model, tier=tier,
                        with_logprobs=n_with_lp, total=len(recs),
                        note="cell falls back to COUNT weighting so one calibrator is not fit "
                             "over two scales; delete the sample checkpoint to regenerate")
            cell_cfg = cfg if cell_lp_ok else replace(cfg, ENTROPY_LP_WEIGHTED=False)

            for r in tqdm(recs, desc=f"entropy {model[:14]}|{tier}", leave=False):
                q = gold.get(r["qid"])
                if q is None:
                    continue
                parsed = r["parsed"] if isinstance(r["parsed"], list) else [r["parsed"]]
                raw_lp = r.get("seq_logprob")
                keep = [i for i, pp in enumerate(parsed) if pp.get("answer")]
                answers = [parsed[i]["answer"] for i in keep]
                lps = ([raw_lp[i] for i in keep]
                       if isinstance(raw_lp, list) and len(raw_lp) == len(parsed) else None)
                n = len(answers)
                low_valid = n < int(cfg.ENTROPY_MIN_VALID)
                if n == 0:
                    rows.append(dict(model=model, tier=tier, qid=r["qid"], split=r["split"],
                                     n_valid=0, n_requested=n_req, n_clusters=0, entropy=np.nan,
                                     entropy_max=H_max_fixed, confidence_behavioral=np.nan,
                                     modal_share=np.nan, modal_mass=np.nan,
                                     low_valid=True, lp_weighted=False))
                    continue
                labels = cluster_answers(answers, q["answer_form"], cell_cfg, nli,
                                         question=q.get("question", ""))
                sizes, lp_weighted = cluster_mass(labels, lps, cell_cfg)
                p = sizes / sizes.sum()
                H = float(sps.entropy(p))
                conf = float(1 - H / H_max_fixed) if H_max_fixed > 0 else np.nan
                counts = np.array([sum(1 for l in labels if l == k) for k in sorted(set(labels))],
                                  dtype=float)
                # A question below the valid-sample floor gets no dispersion
                # estimate at all — and none of the summary statistics that
                # would otherwise read as certainty. One parsed sample used to
                # record entropy 0.0 / modal_share 1.0, which is the same
                # artefact in a different column.
                rows.append(dict(
                    model=model, tier=tier, qid=r["qid"], split=r["split"], n_valid=n,
                    n_requested=n_req,
                    n_clusters=(np.nan if low_valid else int(len(sizes))),
                    entropy=(np.nan if low_valid else H),
                    entropy_max=H_max_fixed,
                    confidence_behavioral=(np.nan if low_valid else conf),
                    modal_share=(np.nan if low_valid else float(counts.max() / counts.sum())),
                    modal_mass=(np.nan if low_valid else float(sizes.max() / sizes.sum())),
                    low_valid=bool(low_valid), lp_weighted=bool(lp_weighted),
                ))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_parquet(PATHS["derived"] / "entropy.parquet", index=False)
    LOG.log("entropy", rows=len(df),
            mean_conf=round(float(df["confidence_behavioral"].mean()), 4) if len(df) else None,
            low_valid_rows=int(df["low_valid"].sum()) if len(df) else 0,
            lp_weighted_rows=int(df["lp_weighted"].sum()) if len(df) else 0,
            min_valid=int(cfg.ENTROPY_MIN_VALID))
    return df


def entropy_sanity_check(cfg: Config) -> dict:
    """PLAN §5·4 positive control: all-same -> H=0 -> conf=1; even split -> conf≈0."""
    n = cfg.N_SAMPLES
    same = np.array([float(n)])
    even = np.ones(n)
    def conf(sizes):
        p = sizes / sizes.sum()
        H = float(sps.entropy(p))
        return 1 - H / np.log(n) if n > 1 else 1.0
    res = {"all_same_confidence": round(conf(same), 6), "even_split_confidence": round(conf(even), 6),
           "n_samples": n}
    res["passes"] = bool(abs(res["all_same_confidence"] - 1.0) < 1e-9 and res["even_split_confidence"] < 1e-9)
    json_write(PATHS["derived"] / "entropy_sanity.json", res)
    LOG.log("entropy_sanity", **res)
    return res
# %%
# ============================================================================
# CELL 14 — probes: 5-percentile sweep, nulls, Gate 3 (PLAN §6, §14.1, §16)
# ============================================================================
from sklearn.feature_extraction.text import TfidfVectorizer      # noqa: E402
from sklearn.isotonic import IsotonicRegression                  # noqa: E402
from sklearn.linear_model import LogisticRegression              # noqa: E402
from sklearn.metrics import roc_auc_score                        # noqa: E402
from sklearn.pipeline import make_pipeline                       # noqa: E402
from sklearn.preprocessing import StandardScaler                 # noqa: E402


def load_activations(model: str, tier: str) -> tuple[list[str], dict[int, np.ndarray]] | None:
    path = PATHS["acts"] / f"{model}__{tier}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    qids = [str(q) for q in z["qids"]]
    mats = {int(k[1:]): z[k] for k in z.files if k.startswith("p")}
    return qids, mats


def probe_labels(model: str, tier: str, qids: list[str], graded: pd.DataFrame,
                  entropy: pd.DataFrame, cfg: Config,
                  split_of: "dict | None" = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y, mask). Binary correctness or thresholded semantic entropy."""
    if cfg.PROBE_LABEL == "entropy" and len(entropy):
        sub = entropy[(entropy.model == model) & (entropy.tier == tier)].set_index("qid")
        vals = np.array([sub["confidence_behavioral"].get(q, np.nan) for q in qids], dtype=float)
        mask = np.isfinite(vals)
        # The median that defines the split is a fitted quantity, so it comes
        # from TRAIN rows only. Taking it over train+cal+test lets the test
        # split influence its own labels — a leak that matters more now that
        # "entropy" is the default probe label.
        tr_mask = mask & np.array([split_of.get(q) == "train" for q in qids]) \
            if split_of else mask
        src_vals = vals[tr_mask] if tr_mask.any() else vals[mask]
        med = float(np.nanmedian(src_vals)) if src_vals.size else 0.5
        return (vals >= med).astype(int), mask
    # Label priority. EXTRACT is the greedy pass whose activations were tapped,
    # so it is the only variant whose correctness the probe is actually being
    # asked to predict (PLAN §6·1). FORCED is the next-closest single greedy
    # answer. SAMPLE is a T>0 draw and is a last resort: using it means the
    # probe is trained on "was one stochastic sample right", which is label
    # noise relative to the activation it is paired with.
    cell = graded[(graded.model == model) & (graded.tier == tier)]
    sub = cell[cell.variant == "EXTRACT"]
    source = "EXTRACT"
    if not len(sub):
        sub = cell[cell.variant == "FORCED"]
        source = "FORCED"
    if not len(sub):
        sub = cell[(cell.variant == "SAMPLE") & (cell.sample_idx == 0)]
        source = "SAMPLE(noisy)"
    if source != "EXTRACT":
        LOG.log("probe_label_fallback", cell=cell_id(model, tier), using=source)
    lut = dict(zip(sub["qid"], sub["correct"].astype(int)))
    vals = np.array([lut.get(q, -1) for q in qids])
    return np.clip(vals, 0, 1), vals >= 0


def fit_probe(X: np.ndarray, y: np.ndarray, cfg: Config, seed: int, C: float = 1.0) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=cfg.PROBE_MAX_ITER, C=float(C), random_state=seed, n_jobs=None),
    ).fit(X, y)


def select_probe_C(X_tr: np.ndarray, y_tr: np.ndarray, cfg: Config,
                    seed: int, n_folds: int = 3) -> tuple[float, dict]:
    """Choose the logistic regularisation strength by cross-validation INSIDE
    the training split.

    `PROBE_C_GRID` was declared and never read: every probe was fit at C=1.0,
    which on a ~3584-dimensional activation with a few hundred training rows is
    barely regularised — hence `auroc_train = 1.000` in 51 of 65 rows (AUDIT
    finding 8).

    The obvious repair — pick C by calibration AUROC — quietly breaks something
    else. Calibration AUROC is not just a selection statistic here: the same
    number drives `meets_gate` (>= AUROC_GATE), the winning-percentile choice,
    and Gate 3. Maximising it over four values of C and then gating on the
    maximum is selection on the test statistic. Measured on pure noise
    (200 trials, n=160, d=40), it moved mean calibration AUROC 0.5046 -> 0.5234
    and the false-pass rate at the 0.65 gate from 1.5% to 4.0%.

    So selection happens inside TRAIN, by k-fold CV, and calibration is left
    untouched for the roles PLAN §6·6 gives it. Ties break toward the SMALLER C
    (stronger regularisation), so an uninformative sweep lands on the most
    constrained probe rather than the most flexible one.
    """
    if not cfg.PROBE_C_SELECT or not cfg.PROBE_C_GRID:
        return 1.0, {"selected_C": 1.0, "method": "fixed", "grid": {}, "n_folds": 0}
    y_tr = np.asarray(y_tr)
    classes, counts = np.unique(y_tr, return_counts=True)
    k = int(min(n_folds, counts.min())) if classes.size >= 2 else 0
    if k < 2:
        return 1.0, {"selected_C": 1.0, "method": "too_few_per_class", "grid": {}, "n_folds": 0}
    rng = np.random.default_rng(seed)
    # Stratified folds, so every fold keeps both classes.
    fold = np.empty(len(y_tr), dtype=int)
    for c in classes:
        idx = np.where(y_tr == c)[0]
        fold[rng.permutation(idx)] = np.arange(len(idx)) % k
    scores: dict[float, float] = {}
    for C in cfg.PROBE_C_GRID:
        per_fold = []
        for f in range(k):
            tr, va = fold != f, fold == f
            if len(np.unique(y_tr[tr])) < 2 or len(np.unique(y_tr[va])) < 2:
                continue
            try:
                sc = fit_probe(X_tr[tr], y_tr[tr], cfg, seed, C=C).predict_proba(X_tr[va])[:, 1]
                per_fold.append(safe_auroc(y_tr[va], sc))
            except Exception:                             # noqa: BLE001
                per_fold.append(float("nan"))
        good = [v for v in per_fold if np.isfinite(v)]
        scores[float(C)] = float(np.mean(good)) if good else float("nan")
    finite = {c: v for c, v in scores.items() if np.isfinite(v)}
    if not finite:
        return 1.0, {"selected_C": 1.0, "method": "no_finite_auroc",
                     "grid": {str(k_): None for k_ in scores}, "n_folds": k}
    best = max(finite.values())
    chosen = min(c for c, v in finite.items() if v == best)
    return chosen, {"selected_C": chosen, "method": "train_cv_auroc", "n_folds": k,
                    "grid": {str(k_): (None if not np.isfinite(v) else round(float(v), 4))
                             for k_, v in scores.items()}}


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def stage_probe(bank: dict, graded: pd.DataFrame, entropy: pd.DataFrame,
                committed: dict, cfg: Config) -> pd.DataFrame:
    """One logistic probe per (cell × percentile). Layer selection happens on
    the CALIBRATION split only (PLAN §6·6, §14.2) — never train, never test."""
    rows = []
    gate3_blocked: dict[str, dict] = {}
    split_of = {t: {r["qid"]: r["split"] for r in bank[t]} for t in bank}
    qtext = {t: {r["qid"]: r["question"] for r in bank[t]} for t in bank}

    for cid, v in committed.items():
        if not v["committed"]:
            continue
        model, tier = v["model"], v["tier"]
        loaded = load_activations(model, tier)
        if loaded is None:
            continue
        qids, mats = loaded
        splits = np.array([split_of[tier].get(q, "train") for q in qids])
        y, mask = probe_labels(model, tier, qids, graded, entropy, cfg,
                               split_of=split_of.get(tier, {}))
        fin = json_read(PATHS["acts"] / f"{model}__{tier}.finiteness.json", {})

        tr = mask & (splits == "train")
        ca = mask & (splits == "calibration")
        te = mask & (splits == "test")
        if tr.sum() < 20 or ca.sum() < 10:
            LOG.log("probe_skipped", cell=cid, n_train=int(tr.sum()), n_cal=int(ca.sum()))
            continue

        # Gate 3 is a PRE-check, not a post-hoc note. It used to be computed
        # and then ignored, so probes were fitted on activations already known
        # to contain NaN/Inf and the resulting AUROC was reported alongside
        # clean cells (AUDIT finding 8). Block on both the stored finiteness
        # record and the arrays actually in hand — a stale or missing
        # .finiteness.json must not buy a dirty cell a free pass.
        if cfg.GATE3_ENFORCE:
            # Element-level, to match the `nonfinite_frac` the finiteness
            # record stores; row counts are reported alongside so a single bad
            # row is distinguishable from a systematically corrupt shard.
            stored_bad = max((v.get("nonfinite_frac", 0.0) for v in fin.values()), default=0.0)
            runtime_bad = max((float(1.0 - np.isfinite(m).mean()) if m.size else 0.0)
                              for m in mats.values()) if mats else 0.0
            bad_rows = {f"p{k}": int((~np.isfinite(m).all(axis=1)).sum())
                        for k, m in mats.items() if m.size}
            if stored_bad > 0.0 or runtime_bad > 0.0:
                gate3_blocked[cid] = {"stored_nonfinite_frac": float(stored_bad),
                                      "runtime_nonfinite_frac": float(runtime_bad),
                                      "nonfinite_rows_by_pct": bad_rows,
                                      "n_rows": int(len(qids))}
                LOG.log("gate3_block", cell=cid, stored=round(stored_bad, 8),
                        runtime=round(runtime_bad, 8), nonfinite_rows=bad_rows,
                        note="dirty activations — probe not fitted (Gate 3). "
                             "Set GATE3_ENFORCE=False to fit anyway and inspect.")
                continue

        for pct in cfg.PERCENTILES:
            X = mats.get(pct)
            if X is None:
                continue
            finite_rows = np.isfinite(X).all(axis=1)
            tr_p, ca_p, te_p = tr & finite_rows, ca & finite_rows, te & finite_rows
            if tr_p.sum() < 20 or ca_p.sum() < 10 or len(np.unique(y[tr_p])) < 2:
                continue

            C_sel, C_meta = select_probe_C(X[tr_p], y[tr_p], cfg, cfg.SEED)
            clf = fit_probe(X[tr_p], y[tr_p], cfg, cfg.SEED, C=C_sel)
            s_tr = clf.predict_proba(X[tr_p])[:, 1]
            s_ca = clf.predict_proba(X[ca_p])[:, 1]
            s_te = clf.predict_proba(X[te_p])[:, 1] if te_p.sum() else np.array([])

            # Label-shuffle null (PLAN §14.1)
            rng = np.random.default_rng(cfg.SEED + pct)
            null = []
            for _ in range(cfg.LABEL_SHUFFLE_REPEATS):
                yp = rng.permutation(y[tr_p])
                if len(np.unique(yp)) < 2:
                    continue
                # Refit at the SELECTED C: a null fitted at a different
                # regularisation strength is not a null for this probe.
                null.append(safe_auroc(y[ca_p], fit_probe(X[tr_p], yp, cfg, cfg.SEED, C=C_sel)
                                       .predict_proba(X[ca_p])[:, 1]))
            null = np.array([x for x in null if np.isfinite(x)])

            # Surface / prompt-only baseline (PLAN §14.1, §17.3)
            auroc_surface = float("nan")
            if cfg.SURFACE_BASELINE:
                try:
                    texts = [qtext[tier].get(q, "") for q in qids]
                    vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2)
                    Xs = vec.fit_transform([texts[i] for i in np.where(tr_p)[0]])
                    sclf = LogisticRegression(max_iter=1000, random_state=cfg.SEED).fit(Xs, y[tr_p])
                    Xc = vec.transform([texts[i] for i in np.where(ca_p)[0]])
                    auroc_surface = safe_auroc(y[ca_p], sclf.predict_proba(Xc)[:, 1])
                except Exception:                          # noqa: BLE001
                    pass

            auroc_cal = safe_auroc(y[ca_p], s_ca)
            rows.append(dict(
                cell=cid, model=model, tier=tier, family=TIER_SPECS[tier]["family"],
                params_b=MODEL_SPECS[model]["params_b"], layer_pct=pct,
                layer_index=percentile_layers(MODEL_SPECS[model]["layers"], [pct])[pct],
                n_train=int(tr_p.sum()), n_cal=int(ca_p.sum()), n_test=int(te_p.sum()),
                base_rate=float(y[tr_p].mean()),
                auroc_train=safe_auroc(y[tr_p], s_tr), auroc_cal=auroc_cal,
                auroc_test=safe_auroc(y[te_p], s_te) if te_p.sum() else float("nan"),
                auroc_null_mean=float(null.mean()) if null.size else float("nan"),
                auroc_null_p95=float(np.percentile(null, 95)) if null.size else float("nan"),
                beats_null=bool(null.size and auroc_cal > np.percentile(null, 95)),
                auroc_surface=auroc_surface,
                beats_surface=bool(np.isfinite(auroc_surface) and auroc_cal > auroc_surface),
                nonfinite_frac=fin.get(f"p{pct}", {}).get("nonfinite_frac", 0.0),
                meets_gate=bool(np.isfinite(auroc_cal) and auroc_cal >= cfg.AUROC_GATE),
                probe_C=float(C_sel), probe_C_method=C_meta["method"],
                probe_C_grid=json.dumps(C_meta["grid"]),
            ))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_parquet(PATHS["derived"] / "probe_sweep.parquet", index=False)
    json_write(PATHS["derived"] / "gate3_blocked_cells.json", gate3_blocked)
    LOG.log("probe_sweep", rows=len(df), cells=int(df["cell"].nunique()) if len(df) else 0,
            gate3_blocked=len(gate3_blocked))
    return df


def gate3_verdict(sweep: pd.DataFrame, cfg: Config, blocked: "dict | None" = None) -> dict:
    """Per-cell: finiteness pre-check, the p0 negative control, then >=1
    percentile with AUROC >= gate that also beats the shuffle null and the
    surface baseline.

    The p0 control is the load-bearing one. Percentile 0 is the output of
    `embed_tokens` at the LAST PROMPT TOKEN — under every prompt template in
    PLAN §4/§6 that position is a fixed template suffix, and a token embedding
    carries no position or context, so p0 is the SAME VECTOR for every question
    in a cell. A probe on a constant cannot separate anything: AUROC must be
    ~0.50. The pre-audit tap scored 0.717 there, which is exactly how the
    audit established it was reading a generated token rather than a prompt
    token. Making this blocking means that class of bug can never again reach
    a results table unnoticed.
    """
    out = {}
    blocked = blocked or {}
    tol = float(cfg.P0_NEG_CONTROL_TOL)
    for cell, g in (sweep.groupby("cell") if len(sweep) else []):
        worst_finite = float(g["nonfinite_frac"].max())
        best = g.loc[g["auroc_cal"].idxmax()] if g["auroc_cal"].notna().any() else None

        p0 = g[g["layer_pct"] == 0]
        p0_auroc = float(p0["auroc_cal"].iloc[0]) if len(p0) and np.isfinite(p0["auroc_cal"].iloc[0]) else float("nan")
        p0_dev = abs(p0_auroc - 0.5) if np.isfinite(p0_auroc) else float("nan")
        # Absent p0 cannot certify the tap; treat as not-passed, not as passed.
        p0_ok = bool(np.isfinite(p0_dev) and p0_dev <= tol)

        signal_ok = bool(best is not None and best["meets_gate"] and best["beats_null"]
                         and (best["beats_surface"] or not cfg.SURFACE_BASELINE))
        clean = worst_finite == 0.0
        passes = bool(signal_ok and p0_ok and clean)
        if not clean:
            diagnosis = "dirty activations"
        elif not p0_ok:
            diagnosis = ("p0 negative control FAILED — the tap is not reading a constant "
                         "prompt-ending token (see AUDIT finding 1)" if np.isfinite(p0_dev)
                         else "p0 negative control unavailable — no p0 probe was fitted")
        elif not signal_ok:
            diagnosis = "clean activations, no signal"
        else:
            diagnosis = "pass"
        out[cell] = {
            "activations_clean": clean, "worst_nonfinite_frac": worst_finite,
            "best_layer_pct": int(best["layer_pct"]) if best is not None else None,
            "best_auroc_cal": float(best["auroc_cal"]) if best is not None else None,
            "beats_null": bool(best["beats_null"]) if best is not None else False,
            "beats_surface": bool(best["beats_surface"]) if best is not None else False,
            "probe_C": (float(pd.to_numeric(best["probe_C"], errors="coerce"))
                        if best is not None and "probe_C" in best.index
                        and np.isfinite(pd.to_numeric(best["probe_C"], errors="coerce"))
                        else None),
            "p0_neg_control_auroc": None if not np.isfinite(p0_auroc) else p0_auroc,
            "p0_neg_control_dev": None if not np.isfinite(p0_dev) else float(p0_dev),
            "p0_neg_control_tol": tol,
            "p0_neg_control_pass": p0_ok,
            "signal_ok": signal_ok,
            "passes": passes,
            "diagnosis": diagnosis,
        }
    for cell, info in blocked.items():
        out.setdefault(cell, {
            "activations_clean": False, "worst_nonfinite_frac": info.get("stored_nonfinite_frac", 1.0),
            "best_layer_pct": None, "best_auroc_cal": None, "beats_null": False,
            "beats_surface": False, "probe_C": None,
            "p0_neg_control_auroc": None, "p0_neg_control_dev": None,
            "p0_neg_control_tol": tol, "p0_neg_control_pass": False, "signal_ok": False,
            "passes": False,
            "diagnosis": "blocked before fitting — dirty activations (Gate 3 pre-check)",
        })
    json_write(PATHS["derived"] / "gate3.json", out)
    LOG.log("gate3", cells=len(out), passed=sum(v["passes"] for v in out.values()),
            p0_control_failures=sum(1 for v in out.values() if not v["p0_neg_control_pass"]),
            blocked=len(blocked))
    return out


# %%
# ============================================================================
# CELL 15 — calibration + scoring rules (PLAN §8)
# ============================================================================


def fit_calibrator(scores: np.ndarray, labels: np.ndarray, cfg: Config):
    """Isotonic when there is enough calibration data, else Platt (PLAN §8·1)."""
    ok = np.isfinite(scores) & np.isfinite(labels)
    s, y = scores[ok], labels[ok]
    if len(s) < 10 or len(np.unique(y)) < 2:
        return lambda x: np.clip(x, 0, 1), "identity"
    method = cfg.CALIBRATOR
    if method == "auto":
        method = "isotonic" if len(s) >= cfg.ISOTONIC_MIN_N else "platt"
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
        return (lambda x: np.clip(iso.predict(np.asarray(x, dtype=float)), 0, 1)), "isotonic"
    lr = LogisticRegression(max_iter=1000).fit(s.reshape(-1, 1), y)
    return (lambda x: lr.predict_proba(np.asarray(x, dtype=float).reshape(-1, 1))[:, 1]), "platt"


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, bins: int) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def murphy_decomposition(p: np.ndarray, y: np.ndarray, bins: int) -> dict:
    """Brier = reliability - resolution + uncertainty (PLAN §8·3).

    Reported per signal instead of raw Brier, because across cells with
    different base rates 'calibration improved' is otherwise inseparable from
    'accuracy improved'.
    """
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return {k: float("nan") for k in ("brier", "reliability", "resolution", "uncertainty", "n")}
    base = y.mean()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rel = res = 0.0
    n = len(p)
    for b in range(bins):
        m = idx == b
        nb = m.sum()
        if nb == 0:
            continue
        pb, ob = p[m].mean(), y[m].mean()
        rel += nb * (pb - ob) ** 2
        res += nb * (ob - base) ** 2
    return {"brier": brier(p, y), "reliability": rel / n, "resolution": res / n,
            "uncertainty": float(base * (1 - base)), "n": int(n), "base_rate": float(base)}


def bootstrap_ci(fn: Callable[[np.ndarray], float], data: np.ndarray, n_boot: int,
                 ci: float, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(data)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    stats_ = np.array([fn(data[rng.integers(0, n, n)]) for _ in range(n_boot)])
    stats_ = stats_[np.isfinite(stats_)]
    if stats_.size == 0:
        return fn(data), float("nan"), float("nan")
    a = (1 - ci) / 2
    return fn(data), float(np.percentile(stats_, 100 * a)), float(np.percentile(stats_, 100 * (1 - a)))


def bootstrap_diff_ci(x: np.ndarray, y: np.ndarray, stat: Callable, n_boot: int,
                      ci: float, seed: int) -> dict:
    """CI on stat(x) - stat(y) for two independent samples (PLAN §13 H1/H3)."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        a = stat(x[rng.integers(0, len(x), len(x))]) if len(x) else np.nan
        b = stat(y[rng.integers(0, len(y), len(y))]) if len(y) else np.nan
        d.append(a - b)
    d = np.array([v for v in d if np.isfinite(v)])
    if d.size == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "excludes_zero": False}
    a = (1 - ci) / 2
    lo, hi = float(np.percentile(d, 100 * a)), float(np.percentile(d, 100 * (1 - a)))
    point = (stat(x) if len(x) else np.nan) - (stat(y) if len(y) else np.nan)
    return {"delta": float(point), "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}


def spearman_with_ci(a: np.ndarray, b: np.ndarray, n_boot: int, ci: float, seed: int) -> dict:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return {"rho": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(len(a)), "p": float("nan")}
    rho, pval = sps.spearmanr(a, b)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        if np.std(a[i]) == 0 or np.std(b[i]) == 0:
            continue
        boots.append(sps.spearmanr(a[i], b[i]).statistic)
    boots = np.array([x for x in boots if np.isfinite(x)])
    q = (1 - ci) / 2
    return {"rho": float(rho), "p": float(pval), "n": int(len(a)),
            "lo": float(np.percentile(boots, 100 * q)) if boots.size else float("nan"),
            "hi": float(np.percentile(boots, 100 * (1 - q))) if boots.size else float("nan")}
# %%
# ============================================================================
# CELL 16 — signal assembly: one row per (model, tier, qid) with all 3 signals
# ============================================================================

SIGNALS = ("verbal", "behavioral", "internal")

# ---------------------------------------------------------------------------
# Epistemic-alignment taxonomy for the verbal x behavioral 2x2.
#
# "Hopeful confidence" anthropomorphises — it attributes a wish to a model —
# and the audit is right that it will draw fire at an NLP venue. These names
# describe the OBSERVED RELATION between two measurements and claim nothing
# about the model's inner life. Storage keys keep the legacy strings so old
# artefacts, figures and downstream notebooks keep loading; only the reported
# labels change.
# ---------------------------------------------------------------------------
QUADRANT_TAXONOMY: dict[str, dict] = {
    "hopeful": {
        "label": "Performative Certainty",
        "verbal": "high", "behavioral": "low",
        "definition": "Stated confidence exceeds sample stability.",
        "neutral_phrase": "unwarranted stated confidence",
    },
    "suppressed": {
        "label": "Excessive Hedging",
        "verbal": "low", "behavioral": "high",
        "definition": "Stated confidence under-reports unanimous sample consensus.",
        "neutral_phrase": "understated stated confidence",
    },
    "agree_high": {
        "label": "Grounded Certainty",
        "verbal": "high", "behavioral": "high",
        "definition": "Stated confidence and sample stability agree, both high.",
        "neutral_phrase": "congruent high confidence",
    },
    "agree_low": {
        "label": "Honest Doubt",
        "verbal": "low", "behavioral": "low",
        "definition": "Stated confidence and sample stability agree, both low.",
        "neutral_phrase": "congruent low confidence",
    },
}
QUADRANT_LABELS: dict[str, str] = {k: v["label"] for k, v in QUADRANT_TAXONOMY.items()}


def quadrant_label(key: str) -> str:
    """Legacy storage key -> reported taxonomy name."""
    return QUADRANT_LABELS.get(key, key)


def empirical_bucket_map(graded: pd.DataFrame, cfg: Config) -> dict:
    """PLAN §4 / §14.3 manipulation check: bucket -> probability comes from the
    ACTUAL accuracy of answers placed in that bucket, never hand-assigned.
    Fit on the calibration split only."""
    sub = graded[(graded.variant == "B") & graded.bucket.notna() & (graded.split == "calibration")]
    out: dict[str, dict] = {}
    for (model, tier), g in sub.groupby(["model", "tier"]):
        m = {}
        for b in BUCKETS:
            gb = g[g.bucket == b]
            m[b] = {"p": float(gb["correct"].mean()) if len(gb) else np.nan, "n": int(len(gb))}
        # Fall back to the pooled rate for buckets the cell never used.
        pooled = float(g["correct"].mean()) if len(g) else np.nan
        for b in BUCKETS:
            if not np.isfinite(m[b]["p"]):
                m[b]["p"] = pooled
                m[b]["imputed"] = True
        out[cell_id(model, tier)] = m
    json_write(PATHS["derived"] / "bucket_mapping.json", out)
    return out


def verbal_scores(graded: pd.DataFrame, bmap: dict, cfg: Config) -> pd.DataFrame:
    """Convert each format to a common 0–1 scale (PLAN §4 agreement check §2)."""
    rows = []
    g = graded[graded.variant.isin(["A", "B", "C"]) & (graded.sample_idx == 0)]
    for r in g.itertuples():
        cid = cell_id(r.model, r.tier)
        if r.variant == "A":
            # The model's own stated number, parsed /100 upstream. Clipped, not
            # remapped: this is the canonical label-free verbal axis.
            v = np.clip(float(r.confidence), 0.0, 1.0) if r.confidence is not None and np.isfinite(
                pd.to_numeric(r.confidence, errors="coerce")) else np.nan
        elif r.variant == "B":
            v = bmap.get(cid, {}).get(r.bucket, {}).get("p") if r.bucket else np.nan
        else:
            # Format C is a decision, not a probability: an ANSWER means the
            # model judged p(correct) above the rational betting threshold.
            thr = -BET_LOSS / (BET_GAIN - BET_LOSS)          # = 2/3 at +1/-2
            v = (1 + thr) / 2 if r.decision == "ANSWER" else thr / 2 if r.decision == "PASS" else np.nan
        try:
            v = float(v)
            if not np.isfinite(v):
                v = np.nan
        except (TypeError, ValueError):
            v = np.nan
        rows.append(dict(model=r.model, tier=r.tier, qid=r.qid, split=r.split,
                         is_agreement=r.is_agreement, variant=r.variant, verbal_raw=v,
                         correct=int(r.correct), decision=r.decision, bucket=r.bucket))
        if r.variant == "B":
            # Same utterance, label-free reading (PLAN §4·4 / AUDIT finding 2).
            rows.append(dict(model=r.model, tier=r.tier, qid=r.qid, split=r.split,
                             is_agreement=r.is_agreement, variant="Bfix",
                             verbal_raw=float(BUCKET_FIXED_VALUES[r.bucket]) if r.bucket in BUCKET_FIXED_VALUES
                                        else np.nan,
                             correct=int(r.correct), decision=r.decision, bucket=r.bucket))
    return pd.DataFrame(rows)


def internal_scores(sweep: pd.DataFrame, bank: dict, graded: pd.DataFrame,
                     entropy: pd.DataFrame, committed: dict, cfg: Config,
                     gate3: "dict | None" = None) -> pd.DataFrame:
    """Refit the winning-percentile probe and emit per-question test scores.
    Winner selected on CALIBRATION only (PLAN §6·6).

    A cell that fails Gate 3 emits NOTHING. Blocking dirty activations before
    fitting is only half of enforcement: the p0 negative control and the AUROC
    gate were still report-only, so a cell whose p0 scored 0.717 — the exact
    signature of the tap defect in AUDIT finding 1 — was written into
    `gate3.json` as failed and *simultaneously* contributed 200 per-question
    `internal_cal` values to H1, H3, H4, the GLMM and the quadrants. A gate
    nothing downstream honours is a comment.
    """
    rows = []
    if not len(sweep):
        return pd.DataFrame(rows)
    if cfg.GATE3_ENFORCE and not gate3:
        # Fail CLOSED. `gate3=None` used to mean "emit everything", so any call
        # site that forgot the argument silently disabled the whole enforcement
        # path this function exists to provide.
        raise RuntimeError(
            "internal_scores: GATE3_ENFORCE is on but no Gate 3 verdict was supplied. "
            "Pass gate3=gate3_verdict(...), or set GATE3_ENFORCE=False deliberately.")
    gate3 = gate3 or {}
    split_of = {t: {r["qid"]: r["split"] for r in bank[t]} for t in bank}
    skipped: dict[str, str] = {}
    for cell, g in sweep.groupby("cell"):
        verdict = gate3.get(cell)
        if cfg.GATE3_ENFORCE and not (verdict or {}).get("passes"):
            # A cell with no verdict at all is also not a cell that passed.
            skipped[cell] = (verdict or {}).get("diagnosis", "no Gate 3 verdict for this cell")
            continue
        g = g[g["auroc_cal"].notna()]
        if not len(g):
            continue
        best = g.loc[g["auroc_cal"].idxmax()]
        model, tier, pct = best["model"], best["tier"], int(best["layer_pct"])
        loaded = load_activations(model, tier)
        if loaded is None:
            continue
        qids, mats = loaded
        X = mats.get(pct)
        if X is None:
            continue
        splits = np.array([split_of[tier].get(q, "train") for q in qids])
        y, mask = probe_labels(model, tier, qids, graded, entropy, cfg,
                               split_of=split_of.get(tier, {}))
        fin = np.isfinite(X).all(axis=1)
        tr = mask & fin & (splits == "train")
        if tr.sum() < 20 or len(np.unique(y[tr])) < 2:
            continue
        # Refit at the C the sweep selected for this cell/percentile, so the
        # emitted per-question scores come from the same probe the sweep
        # reported — not a silently different C=1.0 model.
        C_sel = float(best["probe_C"]) if "probe_C" in best.index and np.isfinite(
            pd.to_numeric(best["probe_C"], errors="coerce")) else 1.0
        clf = fit_probe(X[tr], y[tr], cfg, cfg.SEED, C=C_sel)
        # Score only the finite rows. `predict_proba` on the full matrix raises
        # "Input X contains NaN" the moment any row is non-finite — which is
        # exactly the situation GATE3_ENFORCE=False exists to let you inspect,
        # so the documented escape hatch used to crash on the case it was for.
        s_all = np.full(len(qids), np.nan)
        if fin.any():
            s_all[fin] = clf.predict_proba(X[fin])[:, 1]
        for i, q in enumerate(qids):
            if not fin[i]:
                continue
            rows.append(dict(model=model, tier=tier, qid=q, split=splits[i],
                             internal_raw=float(s_all[i]), best_layer_pct=pct,
                             probe_C=C_sel))
    if skipped:
        LOG.log("internal_scores_gate3_skipped", cells=sorted(skipped), reasons=skipped,
                note="these cells contribute NO internal signal downstream")
    json_write(PATHS["derived"] / "internal_gate3_skipped.json", skipped)
    return pd.DataFrame(rows)


def assemble_signals(bank: dict, graded: pd.DataFrame, entropy: pd.DataFrame,
                     sweep: pd.DataFrame, committed: dict, cfg: Config,
                     gate3: "dict | None" = None) -> tuple[pd.DataFrame, dict]:
    """Wide table: one row per (model, tier, qid) carrying raw + calibrated
    values for all three signals, plus ground truth. Calibrators fit on the
    calibration split, applied to test (PLAN §8·1–2)."""
    bmap = empirical_bucket_map(graded, cfg)
    verbal = verbal_scores(graded, bmap, cfg)
    internal = internal_scores(sweep, bank, graded, entropy, committed, cfg, gate3=gate3)

    # Canonical verbal format (PLAN §4·4), with the §8·6 pre-flight applied
    # BEFORE selection rather than after.
    #
    # Selecting on ECE alone is unsafe here: format B maps each bucket to that
    # bucket's empirical accuracy, so a model that only ever uses one bucket
    # produces a CONSTANT equal to the base rate — which scores ECE ~= 0 while
    # carrying no information at all. Ranking on ECE therefore rewards exactly
    # the degeneracy §8·6 exists to exclude. Brier is used instead because it
    # decomposes into reliability MINUS resolution, so a constant predictor is
    # penalised by its zero resolution.
    #
    # Format B may never be canonical. Its raw value IS the empirical accuracy
    # of the bucket the model chose, fit on the calibration split and then
    # calibrated again on that same split — a supervised P(correct) estimate,
    # not something the model said (AUDIT finding 2). It is scored here purely
    # as a supervised reference, and excluded from selection AND from the
    # degenerate fallback so it cannot slip in through the back door.
    fmt_stats = {}
    for v in VERBAL_FORMATS:
        s = verbal[(verbal.variant == v) & (verbal.split == "calibration")]
        s = s[s.verbal_raw.notna()]
        nd = int(s.verbal_raw.nunique()) if len(s) else 0
        ok = len(s) >= 20
        label_free = v in LABEL_FREE_FORMATS
        fmt_stats[v] = {
            "n": int(len(s)), "n_distinct": nd,
            "ece": ece(s.verbal_raw.values, s.correct.values, cfg.ECE_BINS) if ok else float("inf"),
            "brier": brier(s.verbal_raw.values, s.correct.values.astype(float)) if ok else float("inf"),
            "label_free": label_free,
            "role": "candidate" if label_free else "supervised_reference",
            "eligible": bool(ok and nd >= cfg.MIN_DISTINCT_VERBAL and label_free),
        }
    eligible = [v for v in LABEL_FREE_FORMATS if fmt_stats[v]["eligible"]]
    if eligible:
        canonical = min(eligible, key=lambda v: fmt_stats[v]["brier"])
    else:
        # Nothing clears the pre-flight: keep the most-varied LABEL-FREE format
        # so the pipeline still runs, but the degeneracy is recorded loudly.
        canonical = max(LABEL_FREE_FORMATS, key=lambda v: fmt_stats[v]["n_distinct"])
        LOG.log("verbal_all_formats_degenerate", chosen=canonical,
                n_distinct={k: v["n_distinct"] for k, v in fmt_stats.items()})
    if canonical not in LABEL_FREE_FORMATS:                      # not an assert: -O strips those
        raise RuntimeError(
            f"label-derived format {canonical!r} was selected as the canonical verbal axis; "
            f"only {LABEL_FREE_FORMATS} may be canonical (AUDIT finding 2)")
    fmt_ece = {k: v["ece"] for k, v in fmt_stats.items()}
    LOG.log("canonical_format", chosen=canonical, eligible=eligible,
            supervised_reference=[v for v in VERBAL_FORMATS if v not in LABEL_FREE_FORMATS],
            brier={k: round(v["brier"], 4) for k, v in fmt_stats.items()},
            ece={k: round(v["ece"], 4) for k, v in fmt_stats.items()},
            n_distinct={k: v["n_distinct"] for k, v in fmt_stats.items()})

    v_can = verbal[verbal.variant == canonical][["model", "tier", "qid", "split", "verbal_raw", "correct"]]
    base = v_can.copy()

    # ---- one ground truth for all three signals ------------------------
    gt_variant = cfg.GROUND_TRUTH_VARIANT
    acc_by_variant = {str(v): float(g["correct"].mean())
                      for v, g in graded[graded.sample_idx == 0].groupby("variant")
                      if len(g)}
    spread = (max(acc_by_variant.values()) - min(acc_by_variant.values())) if acc_by_variant else float("nan")
    gt_source = f"canonical_verbal({canonical})"
    if gt_variant != "canonical_verbal":
        gt_all = graded[(graded.variant == gt_variant) & (graded.sample_idx == 0)][
            ["model", "tier", "qid", "correct"]]
        # De-duplicating silently would turn a data-integrity conflict — the
        # same question graded two different ways — into an unrecorded coin
        # flip. Conflicts are counted and named before anything is dropped.
        conflicts = (gt_all.groupby(["model", "tier", "qid"])["correct"].nunique()
                     .pipe(lambda x: x[x > 1]))
        if len(conflicts):
            LOG.log("ground_truth_conflicts", n=int(len(conflicts)),
                    examples=[list(map(str, k)) for k in list(conflicts.index)[:10]],
                    note=f"{gt_variant} grades the same question inconsistently; keeping the "
                         f"last row per key")
        n_dupe = int(len(gt_all) - len(gt_all.drop_duplicates(subset=["model", "tier", "qid"])))
        if n_dupe:
            LOG.log("ground_truth_duplicates", rows_dropped=n_dupe, variant=gt_variant)
        gt = (gt_all.drop_duplicates(subset=["model", "tier", "qid"], keep="last")
              .rename(columns={"correct": "correct_gt"}))
        if len(gt):
            n_before = len(base)
            # `validate` turns a duplicated key into an exception instead of a
            # silently longer table: an unguarded left merge on a duplicated
            # (model, tier, qid) multiplies rows, and every per-cell mean and
            # bootstrap downstream would be computed over the inflated frame.
            base = base.merge(gt, on=["model", "tier", "qid"], how="left", validate="m:1")
            if len(base) != n_before:
                raise RuntimeError(
                    f"ground-truth merge changed the row count {n_before} -> {len(base)}; "
                    f"variant {gt_variant} has duplicate (model, tier, qid) keys")
            n_missing = int(base["correct_gt"].isna().sum())
            # Fall back per-row rather than silently dropping questions the
            # ground-truth pass did not answer; the count is recorded.
            base["correct"] = base["correct_gt"].fillna(base["correct"]).astype(int)
            base = base.drop(columns=["correct_gt"])
            gt_source = gt_variant
            LOG.log("ground_truth_variant", using=gt_variant, rows_backfilled=n_missing,
                    accuracy_by_variant={k: round(v, 4) for k, v in acc_by_variant.items()},
                    max_spread=round(spread, 4) if np.isfinite(spread) else None)
        else:
            LOG.log("ground_truth_variant_missing", requested=gt_variant,
                    note="falling back to the canonical verbal pass's correctness")
    if len(entropy):
        base = base.merge(entropy[["model", "tier", "qid", "confidence_behavioral", "n_clusters", "n_valid"]],
                          on=["model", "tier", "qid"], how="left")
    else:
        base["confidence_behavioral"] = np.nan
    base = base.rename(columns={"confidence_behavioral": "behavioral_raw"})
    if len(internal):
        base = base.merge(internal[["model", "tier", "qid", "internal_raw", "best_layer_pct"]],
                          on=["model", "tier", "qid"], how="left")
    else:
        base["internal_raw"] = np.nan
        base["best_layer_pct"] = np.nan

    # Per (cell x signal) calibration, fit on calibration split only.
    cal_meta: dict[str, dict] = {}
    for sig in SIGNALS:
        base[f"{sig}_cal"] = np.nan
    for (model, tier), g in base.groupby(["model", "tier"]):
        cid = cell_id(model, tier)
        cal_meta[cid] = {}
        idx_cal = g.index[g.split == "calibration"]
        for sig in SIGNALS:
            col = f"{sig}_raw"
            fitted, method = fit_calibrator(g.loc[idx_cal, col].values.astype(float),
                                            g.loc[idx_cal, "correct"].values.astype(float), cfg)
            vals = g[col].values.astype(float)
            ok = np.isfinite(vals)
            out = np.full(len(vals), np.nan)
            if ok.any():
                out[ok] = fitted(vals[ok])
            base.loc[g.index, f"{sig}_cal"] = out
            n_distinct = int(pd.Series(g.loc[idx_cal, col]).nunique(dropna=True))
            cal_meta[cid][sig] = {"method": method, "n_cal": int(len(idx_cal)), "n_distinct": n_distinct,
                                  "excluded": bool(sig == "verbal" and n_distinct < cfg.MIN_DISTINCT_VERBAL)}

    # Enforce PLAN §8·6: a cell whose verbal signal has fewer than
    # MIN_DISTINCT_VERBAL distinct values is EXCLUDED from the verbal
    # comparison — not reported as "well calibrated". Previously this flag was
    # computed and then ignored, so degenerate cells still reached H1/H2/H3.
    n_excl = 0
    for cid, meta_c in cal_meta.items():
        if not meta_c.get("verbal", {}).get("excluded"):
            continue
        model_x, tier_x = cid.split("__")
        mask = (base["model"] == model_x) & (base["tier"] == tier_x)
        base.loc[mask, "verbal_cal"] = np.nan
        base.loc[mask, "verbal_raw"] = np.nan
        n_excl += int(mask.sum())
    base["verbal_excluded"] = base["verbal_cal"].isna()
    # Why an internal value is absent matters: Gate 3 removed the whole cell,
    # or the activation was non-finite, or the cell had too few training rows.
    # Without this they are indistinguishable, and the GLMM's
    # `internal_cal_missing` indicator lumps all three causes together. The
    # band-gate repair replaced deletion with a covariate; Gate 3 must not
    # reintroduce deletion without one.
    _g3_failed = {c for c, v in (gate3 or {}).items() if not v.get("passes")}
    base["internal_excluded_reason"] = np.where(
        base["internal_raw"].notna(), None,
        np.where(base.apply(lambda r: cell_id(r["model"], r["tier"]) in _g3_failed, axis=1),
                 "gate3_failed", "no_probe_score"))
    base["internal_excluded"] = base["internal_raw"].isna()
    LOG.log("verbal_preflight", excluded_cells=sum(
        1 for m in cal_meta.values() if m.get("verbal", {}).get("excluded")),
        total_cells=len(cal_meta), excluded_rows=n_excl)

    # Continuous confidence discrepancy. The quadrant split at 0.5 is a fixed
    # constant, but it is still a knife edge: 4 of the 6 quadrant cells never
    # had verbal_cal cross it at all, and all 224 "hopeful" cases came from
    # three cells whose modal bucket sat 1-4 points above the cut (AUDIT
    # finding 6). Delta carries the same comparison with no cutpoint, so
    # regressions and effect sizes do not inherit the threshold's arbitrariness.
    base["delta_verbal_behavioral"] = (base["verbal_cal"].astype(float)
                                       - base["behavioral_cal"].astype(float))
    base["delta_verbal_internal"] = (base["verbal_cal"].astype(float)
                                     - base["internal_cal"].astype(float))

    base["family"] = base["tier"].map(lambda t: TIER_SPECS[t]["family"])
    base["params_b"] = base["model"].map(lambda m: MODEL_SPECS[m]["params_b"])
    # Base-rate covariates: the band gate no longer deletes cells, so the
    # pilot accuracy it measured travels with the rows instead (AUDIT finding 9).
    # An unknown cell is UNKNOWN, not in-band. Defaulting to True would hand a
    # cell with no pilot record the same covariate value as one measured inside
    # the band.
    base["in_band"] = base.apply(
        lambda r: (committed.get(cell_id(r["model"], r["tier"])) or {}).get("in_band", None), axis=1)
    n_unknown = int(base["in_band"].isna().sum())
    if n_unknown:
        LOG.log("in_band_unknown", rows=n_unknown,
                note="cells with no pilot record — in_band is NaN, not True")
    base["pilot_accuracy"] = base.apply(
        lambda r: float((committed.get(cell_id(r["model"], r["tier"])) or {}).get("accuracy", np.nan)), axis=1)
    base["canonical_format"] = canonical
    base.to_parquet(PATHS["derived"] / "signals.parquet", index=False)
    json_write(PATHS["derived"] / "calibration_meta.json",
               {"canonical_format": canonical, "format_ece": fmt_ece,
                "format_stats": fmt_stats, "cells": cal_meta,
                "label_free_formats": list(LABEL_FREE_FORMATS),
                "ground_truth_variant": gt_source,
                "accuracy_by_variant": acc_by_variant,
                "accuracy_spread_across_variants": (None if not np.isfinite(spread) else float(spread))})
    LOG.log("signals_assembled", rows=len(base), cells=int(base.groupby(["model", "tier"]).ngroups))
    return base, {"canonical_format": canonical, "format_ece": fmt_ece,
                  "format_stats": fmt_stats, "cells": cal_meta,
                  "bucket_map": bmap, "verbal_long": verbal}


# %%
# ============================================================================
# CELL 17 — hypothesis tests H0–H4, gates, quadrants, Omniscience-Index
# ============================================================================


def test_h0_format_agreement(verbal_long: pd.DataFrame, cfg: Config) -> dict:
    """Gate 2 — pairwise Spearman across formats A/B/C on the agreement subset."""
    res = {"pairs": {}, "per_cell": {}}
    sub = verbal_long[verbal_long.is_agreement & verbal_long.verbal_raw.notna()]
    wide = sub.pivot_table(index=["model", "tier", "qid"], columns="variant",
                           values="verbal_raw", aggfunc="first")
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        if a in wide and b in wide:
            m = wide[[a, b]].dropna()
            res["pairs"][f"{a}-{b}"] = spearman_with_ci(m[a].values, m[b].values,
                                                        cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_CI, cfg.SEED)
        else:
            res["pairs"][f"{a}-{b}"] = {"rho": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    for model, gm in sub.groupby("model"):
        w = gm.pivot_table(index=["tier", "qid"], columns="variant", values="verbal_raw", aggfunc="first")
        cell = {}
        for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
            if a in w and b in w:
                m = w[[a, b]].dropna()
                cell[f"{a}-{b}"] = spearman_with_ci(m[a].values, m[b].values, 500, cfg.BOOTSTRAP_CI, cfg.SEED)
        res["per_cell"][model] = cell
    los = [v["lo"] for v in res["pairs"].values() if np.isfinite(v.get("lo", np.nan))]
    res["gate2_pass"] = bool(len(los) == 3 and min(los) >= cfg.GATE2_SPEARMAN)
    res["falsified"] = bool(len(los) == 3 and min(los) < cfg.GATE2_SPEARMAN)
    res["threshold"] = cfg.GATE2_SPEARMAN
    res["verdict"] = ("H0 supported — collapse to canonical format" if res["gate2_pass"]
                      else "H0 falsified — report all three formats separately (PLAN §16 Gate 2 fallback)")
    json_write(PATHS["derived"] / "h0_gate2.json", res)
    LOG.log("gate2", pass_=res["gate2_pass"], **{k: round(v["rho"], 3) for k, v in res["pairs"].items()
                                                  if np.isfinite(v["rho"])})
    return res


def test_h1_signal_calibration(signals: pd.DataFrame, cfg: Config) -> dict:
    """Per-signal ECE/Brier + Murphy on the TEST split, with bootstrap CI on
    Δ(best − worst) (PLAN §13 H1)."""
    out = {"per_cell": {}, "pooled": {}}
    test = signals[signals.split == "test"]

    # H1 compares scoring rules ACROSS signals and bootstraps the best-worst
    # gap, so the comparison is only like-for-like if the signals cover the
    # same questions. They no longer do by default: Gate 3 enforcement drops a
    # failing cell's internal signal entirely, which silently shrinks the
    # internal column's grid. That has to be a reported qualifier on H1, not a
    # log line — a reader comparing ECE across three columns cannot see it
    # otherwise.
    cells_all = {f"{m}__{t}" for m, t in test[["model", "tier"]].drop_duplicates().itertuples(index=False)}
    coverage: dict[str, dict] = {}
    for sig in SIGNALS:
        present = test[test[f"{sig}_cal"].notna()]
        cells_sig = {f"{m}__{t}" for m, t in
                     present[["model", "tier"]].drop_duplicates().itertuples(index=False)}
        coverage[sig] = {"n_rows": int(len(present)), "n_cells": len(cells_sig),
                         "missing_cells": sorted(cells_all - cells_sig)}
    common = set.intersection(*[
        {f"{m}__{t}" for m, t in test[test[f"{s}_cal"].notna()][["model", "tier"]]
         .drop_duplicates().itertuples(index=False)} for s in SIGNALS]) if len(test) else set()
    # Cell coverage is not enough. Row-level missingness — behavioral NaN below
    # ENTROPY_MIN_VALID, internal NaN on a non-finite activation — leaves every
    # signal present in every cell while the three still score different
    # questions. The delta is bootstrapped across those row sets, so the check
    # has to be at row granularity.
    complete = test[[f"{s_}_cal" for s_ in SIGNALS]].notna().all(axis=1) if len(test) else pd.Series(dtype=bool)
    n_complete = int(complete.sum())
    n_rows_by_signal = {s_: coverage[s_]["n_rows"] for s_ in SIGNALS}
    rows_equal = len(set(n_rows_by_signal.values())) <= 1 and n_complete == max(
        n_rows_by_signal.values(), default=0)
    out["coverage"] = {
        "cells_total": len(cells_all), "per_signal": coverage,
        "cells_all_three_signals": sorted(common),
        "cells_equal": bool(len(common) == len(cells_all)),
        "n_rows_by_signal": n_rows_by_signal,
        "n_rows_complete_case": n_complete,
        "n_rows_test": int(len(test)),
        "like_for_like": bool(len(common) == len(cells_all) and rows_equal),
    }
    if not out["coverage"]["like_for_like"]:
        out["coverage"]["warning"] = (
            "Signals do not score the same questions, so pooled ECE/Brier/Murphy and the "
            "best-worst delta are computed over DIFFERENT sets and are not a like-for-like "
            f"comparison. Cells carrying all three: {len(common)} of {len(cells_all)}. "
            f"Rows per signal: {n_rows_by_signal}; only {n_complete} of {len(test)} test rows "
            f"carry all three signals. Use the complete-case subset for any cross-signal claim.")

    for sig in SIGNALS:
        d = test[[f"{sig}_cal", "correct"]].dropna()
        if len(d) < 20:
            out["pooled"][sig] = {"n": len(d)}
            continue
        p, y = d[f"{sig}_cal"].values, d["correct"].values.astype(float)
        pairs = np.column_stack([p, y])
        e, elo, ehi = bootstrap_ci(lambda a: ece(a[:, 0], a[:, 1], cfg.ECE_BINS), pairs,
                                   cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_CI, cfg.SEED)
        b, blo, bhi = bootstrap_ci(lambda a: brier(a[:, 0], a[:, 1]), pairs,
                                   cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_CI, cfg.SEED)
        out["pooled"][sig] = {"n": int(len(d)), "ece": e, "ece_lo": elo, "ece_hi": ehi,
                              "brier": b, "brier_lo": blo, "brier_hi": bhi,
                              **murphy_decomposition(p, y, cfg.MURPHY_BINS)}
    for (model, tier), g in test.groupby(["model", "tier"]):
        cell = {}
        for sig in SIGNALS:
            d = g[[f"{sig}_cal", "correct"]].dropna()
            if len(d) < 20:
                continue
            p, y = d[f"{sig}_cal"].values, d["correct"].values.astype(float)
            cell[sig] = {"ece": ece(p, y, cfg.ECE_BINS), **murphy_decomposition(p, y, cfg.MURPHY_BINS)}
        out["per_cell"][cell_id(model, tier)] = cell

    usable = {s: v for s, v in out["pooled"].items() if "ece" in v}
    if len(usable) >= 2:
        best = min(usable, key=lambda s: usable[s]["ece"])
        worst = max(usable, key=lambda s: usable[s]["ece"])
        out["ece_ranking"] = sorted(usable, key=lambda s: usable[s]["ece"])
        out["resolution_ranking"] = sorted(
            usable, key=lambda s: -usable[s].get("resolution", float("-inf")))
        db = test[[f"{best}_cal", "correct"]].dropna()
        dw = test[[f"{worst}_cal", "correct"]].dropna()
        delta = bootstrap_diff_ci(
            np.column_stack([dw[f"{worst}_cal"].values, dw["correct"].values.astype(float)]),
            np.column_stack([db[f"{best}_cal"].values, db["correct"].values.astype(float)]),
            lambda a: ece(a[:, 0], a[:, 1], cfg.ECE_BINS), cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_CI, cfg.SEED)
        out["delta_worst_minus_best"] = {"worst": worst, "best": best, **delta}
        # H1's STATEMENT is directional: "verbalized confidence runs hot
        # relative to behavioral and internal". A CI on |best - worst| that
        # excludes zero only shows the signals DIFFER — if verbal turns out to
        # be the best-calibrated signal, the ordering contradicts H1 and the
        # hypothesis is falsified, not supported. Both facts are reported.
        separable = bool(delta["excludes_zero"])
        direction_ok = bool(worst == "verbal")
        out["signals_separable"] = separable
        out["direction_matches_prediction"] = direction_ok
        out["h1_pass"] = bool(separable and direction_ok)
        if not separable:
            out["verdict"] = "H1 falsified — CIs overlap, no ordering distinguishable"
        elif direction_ok:
            out["verdict"] = (f"H1 supported — verbal is worse-calibrated than {best} beyond noise")
        else:
            out["verdict"] = (
                f"H1 falsified on direction — signals differ beyond noise, but the ordering is "
                f"reversed: verbal is better calibrated than {worst}. "
                f"ECE rank (best first): {out['ece_ranking']}; "
                f"resolution rank (highest first): {out['resolution_ranking']}. "
                f"Note low verbal ECE with low resolution indicates a near-constant predictor.")
    json_write(PATHS["derived"] / "h1_calibration.json", out)
    LOG.log("h1", pass_=out.get("h1_pass"), **{s: round(v.get("ece", np.nan), 4) for s, v in out["pooled"].items()})
    return out


def question_features(bank: dict) -> pd.DataFrame:
    """Cheap, pre-registered question-level features for the H2 contingency
    test (PLAN §13 H2): length, digits/dates, entity-ish capitalisation."""
    rows = []
    for tier, items in bank.items():
        for r in items:
            q = r["question"]
            rows.append(dict(
                tier=tier, qid=r["qid"],
                n_words=len(q.split()),
                is_long=len(q.split()) > 15,
                has_year=bool(re.search(r"\b(1[6-9]\d\d|20\d\d)\b", q)),
                has_number=bool(re.search(r"\d", q)),
                n_caps=sum(1 for w in q.split()[1:] if w[:1].isupper()),
                has_multi_entity=sum(1 for w in q.split()[1:] if w[:1].isupper()) >= 2,
                family=r["family"],
            ))
    return pd.DataFrame(rows)


def replicates_across_cells(per_cell: dict, min_cells: int = 2, alpha: float = 0.05,
                             p_key: str = "p_conservative") -> dict:
    """A feature association counts as replicated only if it is significant in
    at least `min_cells` independent model x tier cells.

    AUDIT finding 6: H2's only passing features (`is_long`, `family`,
    `has_number`, `has_multi_entity`) are all tier proxies, and the single
    non-tier feature (`has_year`) fails. A pooled chi-square across cells with
    very different base rates will find "associations" that are really just
    cell identity, so a pooled p-value is not evidence about question
    characteristics at all.

    Uses the Yates-corrected p by default. `beats_null` compares against an
    uncorrected permutation null and must stay uncorrected to match it, but a
    claim that a feature GENERALISES should be made on the conservative number.
    """
    def pval(v: dict) -> float:
        x = v.get(p_key, v.get("p", np.nan))
        return float(x) if x is not None and np.isfinite(x) else np.nan
    hits = [c for c, v in per_cell.items() if np.isfinite(pval(v)) and pval(v) < alpha]
    return {"n_cells_tested": len(per_cell), "n_cells_significant": len(hits),
            "significant_cells": sorted(hits), "min_cells_required": min_cells,
            "alpha": alpha, "p_used": p_key,
            "replicates": bool(len(hits) >= min_cells)}


def test_h2_quadrants(signals: pd.DataFrame, bank: dict, cfg: Config,
                      gate1_pass: "bool | None" = None,
                      gate3: "dict | None" = None) -> dict:
    """Quadrant assignment + WITHIN-CELL chi-square vs question features,
    replication across cells, the continuous Delta metric, and the PLAN §16
    gate conjunct that the pass rule used to drop silently."""
    test = signals[signals.split == "test"].copy()
    thr = cfg.QUADRANT_THRESHOLD
    # `other_cal` averages the two non-verbal signals, skipping NaN. That means
    # a row where behavioral confidence is missing silently becomes
    # internal-only — a different quantity under the same column name, on rows
    # that are not a random subset (behavioral is NaN precisely where too few
    # samples parsed). The composition is recorded per row so the axis is never
    # read as homogeneous, and quadrant assignment is restricted to rows where
    # both components exist unless that would empty the table.
    comp = test[["behavioral_cal", "internal_cal"]].notna().sum(axis=1)
    test["other_cal"] = test[["behavioral_cal", "internal_cal"]].mean(axis=1, skipna=True)
    test["other_cal_n_components"] = comp.astype(int)
    n_partial = int(((comp == 1)).sum())
    n_both = int((comp == 2).sum())
    if n_partial and n_both >= 100:
        test = test[comp == 2].copy()
        LOG.log("h2_other_cal_restricted", dropped_partial_rows=n_partial, kept=n_both,
                note="quadrants use rows with BOTH behavioral and internal present")
    elif n_partial:
        LOG.log("h2_other_cal_mixed", partial_rows=n_partial, both_rows=n_both,
                note="too few complete rows to restrict; other_cal mixes one- and "
                     "two-component values — see other_cal_n_components")
    def quad(row):
        v, o = row["verbal_cal"], row["other_cal"]
        if not np.isfinite(v) or not np.isfinite(o):
            return None
        if v >= thr and o < thr:
            return "hopeful"
        if v < thr and o >= thr:
            return "suppressed"
        return "agree_high" if v >= thr else "agree_low"
    test["quadrant"] = test.apply(quad, axis=1)
    test["quadrant_label"] = test["quadrant"].map(lambda k: quadrant_label(k) if k else None)
    feats = question_features(bank)
    merged = test.merge(feats, on=["tier", "qid"], how="left", suffixes=("", "_f"))
    merged = merged[merged.quadrant.notna()]
    merged["cell"] = merged["model"] + "__" + merged["tier"]

    counts = dict(Counter(merged["quadrant"]))
    out = {
        "counts": counts,
        "counts_labelled": {quadrant_label(k): v for k, v in counts.items()},
        "other_cal_rows_both_components": n_both,
        "other_cal_rows_one_component": n_partial,
        "taxonomy": QUADRANT_TAXONOMY,
        "n": int(len(merged)), "threshold": thr,
        "associations": {}, "associations_per_cell": {}, "replication": {},
    }

    # ---- how concentrated is the result? (AUDIT finding 6) --------------
    cell_counts = (merged.groupby(["cell", "quadrant"]).size().unstack(fill_value=0)
                   if len(merged) else pd.DataFrame())
    conc = {}
    for q in ("hopeful", "suppressed"):
        if len(cell_counts) and q in cell_counts:
            s = cell_counts[q].sort_values(ascending=False)
            tot = int(s.sum())
            conc[q] = {
                "total": tot, "n_cells_with_any": int((s > 0).sum()),
                "n_cells_total": int(len(s)),
                "top3_share": float(s.head(3).sum() / tot) if tot else float("nan"),
                "by_cell": {k: int(v) for k, v in s.items() if v > 0},
            }
    out["concentration"] = conc
    # A cell whose verbal_cal never crosses the threshold contributes only one
    # quadrant by construction and is not evidence of anything.
    degenerate = []
    for cell, g in merged.groupby("cell"):
        v = g["verbal_cal"].astype(float)
        if v.notna().any() and (float(v.max()) < thr or float(v.min()) >= thr):
            degenerate.append({"cell": cell, "n": int(len(g)),
                               "verbal_cal_range": [float(v.min()), float(v.max())],
                               "only_quadrant": sorted(set(g["quadrant"]))})
    out["threshold_degenerate_cells"] = degenerate

    # ---- continuous Delta, overall and per quadrant ---------------------
    if "delta_verbal_behavioral" in merged:
        d = merged["delta_verbal_behavioral"].astype(float)
        fin = d[np.isfinite(d)]
        out["delta_verbal_behavioral"] = {
            "n": int(fin.size),
            "mean": float(fin.mean()) if fin.size else float("nan"),
            "sd": float(fin.std(ddof=1)) if fin.size > 1 else float("nan"),
            "quantiles": {str(q): float(fin.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
                          if fin.size else {},
            "by_quadrant": {quadrant_label(k): (float(v) if np.isfinite(v) else None)
                            for k, v in merged.groupby("quadrant")["delta_verbal_behavioral"]
                            .mean().items()},
        }

    FEATURES = ("has_year", "is_long", "has_multi_entity", "has_number", "family")

    def chi2_of(frame: pd.DataFrame, feat: str, n_null: int, tag: str) -> "dict | None":
        """Shuffle-null chi-square, reproducible independently of call order.

        A single shared RNG consumed feature-by-feature and cell-by-cell made
        `null_p95` depend on how many draws earlier tests had taken and on the
        row order of the frame — measured range 6.63-8.94 for the same table
        under the same seed. `beats_null` gates H2's pass rule, so that is a
        verdict that moves when nothing about the data does. The seed is now
        derived from the test's own identity, and the frame is sorted before
        permuting.
        """
        frame = frame.sort_values(["model", "tier", "qid"], kind="mergesort")
        tab = pd.crosstab(frame["quadrant"], frame[feat])
        if tab.shape[0] < 2 or tab.shape[1] < 2 or tab.values.sum() < 20:
            return None
        # correction=False is load-bearing. scipy applies Yates' continuity
        # correction BY DEFAULT and ONLY on 2x2 tables, while the permutation
        # null below is uncorrected — so on a 2x2 the observed statistic would
        # be shrunk and the null would not. Measured shrink: 19.2% at n=60,
        # 17.4% at n=120, 5.8% at n=300 — worst in exactly the per-cell
        # replication tests, where cells carry a couple of hundred test rows.
        # `beats_null` gates H2's pass rule, so the bias runs toward declaring
        # no association. Both sides are now uncorrected and comparable.
        chi2, pv, dof, _ = sps.chi2_contingency(tab, correction=False)
        # A second, CONSERVATIVE p for the replication decision. Dropping Yates
        # is right for `beats_null` — the permutation null is uncorrected, so
        # both sides must be — but `replicates_across_cells` keys off an
        # ANALYTIC p on per-cell 2x2 tables, where the uncorrected version is
        # anti-conservative: measured 0.1013 (Yates) vs 0.0405 (uncorrected) on
        # a sparse 2x2, straddling the 0.05 line. Replication is a claim about
        # generality, so it uses the stricter number.
        pv_conservative = float(sps.chi2_contingency(tab)[1])
        seed = (cfg.SEED + int(hashlib.sha256(f"{tag}|{feat}".encode()).hexdigest()[:8], 16)) % (2**32)
        rng_local = np.random.default_rng(seed)

        # Integer-coded contingency counts via bincount, rather than rebuilding
        # a DataFrame per draw. Same statistic, ~2 orders of magnitude cheaper,
        # which is what makes a null large enough to be stable affordable.
        qcode, _qk = pd.factorize(frame["quadrant"], sort=True)
        fcode, _fk = pd.factorize(frame[feat], sort=True)
        R, C = int(qcode.max()) + 1, int(fcode.max()) + 1
        n_tot = len(qcode)
        row_tot = np.bincount(qcode, minlength=R).astype(float)

        def chi2_from(fc: np.ndarray) -> float:
            obs = np.bincount(qcode * C + fc, minlength=R * C).reshape(R, C).astype(float)
            col_tot = obs.sum(axis=0)
            exp = np.outer(row_tot, col_tot) / n_tot
            nz = exp > 0
            return float((((obs - exp) ** 2)[nz] / exp[nz]).sum())

        nulls = np.empty(n_null, dtype=float)
        for i in range(n_null):
            nulls[i] = chi2_from(rng_local.permutation(fcode))
        p95 = float(np.percentile(nulls, 95)) if n_null else float("nan")
        # The null's MEAN is the stable diagnostic. On a 2x2 the permutation
        # p95 sits on a lumpy discrete distribution and moves between adjacent
        # atoms from draw to draw, while the mean is ~dof under the
        # uncorrected statistic and materially lower under a corrected one —
        # so this is what shows, at a glance, that observed and null are on the
        # same convention.
        null_mean = float(nulls.mean()) if n_null else float("nan")
        return {"chi2": float(chi2), "p": float(pv),
                "p_conservative": pv_conservative,
                "dof": int(dof), "n": int(tab.values.sum()),
                "yates_correction": False, "n_null_draws": int(n_null),
                "null_p95": p95, "null_mean": null_mean,
                "beats_null": bool(n_null and chi2 > p95),
                "table": tab.to_dict()}

    for feat in FEATURES:
        if feat not in merged:
            continue
        pooled = chi2_of(merged, feat, 2000, tag="pooled")
        if pooled is not None:
            out["associations"][feat] = pooled
        # WITHIN-CELL: the same test run inside each cell, where tier and model
        # are held constant and a tier proxy can no longer masquerade as a
        # question characteristic.
        per_cell = {}
        for cell, g in merged.groupby("cell"):
            if len(g) < 40:
                continue
            r = chi2_of(g, feat, 1000, tag=f"cell:{cell}")
            if r is not None:
                per_cell[cell] = r
        out["associations_per_cell"][feat] = per_cell
        out["replication"][feat] = replicates_across_cells(per_cell)

    # ---- pass rule: association AND replication AND the PLAN §16 gates ---
    pooled_sig = [f for f, v in out["associations"].items() if v["p"] < 0.05 and v["beats_null"]]
    replicated = [f for f, v in out["replication"].items() if v["replicates"]]
    out["pooled_significant_features"] = pooled_sig
    out["replicated_features"] = replicated

    gate3 = gate3 or {}
    # Per-cell, not grid-wide. `any(...)` let ONE clean cell out of thirteen
    # satisfy "probe validity" for every other cell in the table, including
    # cells whose p0 negative control had failed.
    cells_used = sorted(set(merged["cell"])) if len(merged) else []
    gate3_failing = [c for c in cells_used if not (gate3.get(c) or {}).get("passes")]
    gate3_ok = bool(gate3) and bool(cells_used) and not gate3_failing
    out["gate3_cells_used"] = cells_used
    out["gate3_cells_failing"] = gate3_failing
    # PLAN §16 makes H2 conditional on "AND Gate 1 holds". The conjunct was
    # computed and then omitted from the pass rule, so H2 could be declared
    # supported while Gate 1 was null in the shipped run (AUDIT finding 8).
    conjuncts = {
        "association_beats_null": bool(pooled_sig),
        "replicates_across_cells": bool(replicated),
        "gate1_grading_sanity": (None if gate1_pass is None else bool(gate1_pass)),
        "gate3_probe_validity": gate3_ok,
    }
    out["conjuncts"] = conjuncts
    failing = [k for k, v in conjuncts.items() if not v]
    out["h2_pass"] = bool(not failing)
    if out["h2_pass"]:
        out["verdict"] = ("H2 supported - quadrant membership associates with question features "
                          "beyond the shuffle null, replicates in >=2 cells, and both grading "
                          "sanity (Gate 1) and probe validity (Gate 3) hold")
    else:
        out["verdict"] = ("H2 not supported - failing conjunct(s): " + ", ".join(failing) +
                          ". Reported as a DESCRIPTIVE result: confidence disagreement is "
                          "heterogeneous across model x tier cells, and the features that "
                          "survive pooling are tier proxies rather than question characteristics.")
    out["reframing"] = ("Descriptive claim only: cross-signal confidence disagreement varies by "
                        "model x tier cell. No general claim about question characteristics is "
                        "made unless `replicated_features` is non-empty.")

    examples = {}
    qtext = {r["qid"]: r["question"] for t in bank for r in bank[t]}
    for q in QUADRANT_TAXONOMY:
        sel = merged[merged.quadrant == q].head(25)
        examples[q] = [{"model": r.model, "tier": r.tier, "qid": r.qid,
                        "quadrant_label": quadrant_label(q),
                        "question": qtext.get(r.qid, ""), "verbal": round(float(r.verbal_cal), 3),
                        "other": round(float(r.other_cal), 3), "correct": int(r.correct)}
                       for r in sel.itertuples()]
    out["examples"] = examples
    merged.to_parquet(PATHS["derived"] / "quadrants.parquet", index=False)
    json_write(PATHS["derived"] / "h2_quadrants.json", out)
    LOG.log("h2", pass_=out["h2_pass"], failing_conjuncts=failing,
            replicated=replicated, **counts)
    return out


def abstention_split(graded: pd.DataFrame, cfg: Config) -> dict:
    """PLAN §4.1 — every Format C Pass becomes justified hedge or missed knowledge."""
    passes = graded[(graded.variant == "C") & (graded.decision == "PASS")][["model", "tier", "qid", "split"]]
    forced = graded[graded.variant == "FORCED"][["model", "tier", "qid", "correct"]]
    m = passes.merge(forced, on=["model", "tier", "qid"], how="left")
    m["category"] = np.where(m["correct"].isna(), "unresolved",
                             np.where(m["correct"] == 1, "missed_knowledge", "justified_hedge"))
    out = {"total_passes": int(len(m)), "by_category": dict(Counter(m["category"])), "per_cell": {}}
    for (model, tier), g in m.groupby(["model", "tier"]):
        c = Counter(g["category"])
        n = len(g)
        out["per_cell"][cell_id(model, tier)] = {
            "n_passes": n, **{k: int(v) for k, v in c.items()},
            "missed_knowledge_rate": round(c.get("missed_knowledge", 0) / n, 4) if n else 0.0,
        }
    out["caveat"] = ("Forced answers are accuracy under compulsion, not ground truth about the "
                     "original hedge (PLAN §4.1).")
    m.to_parquet(PATHS["derived"] / "abstention_split.parquet", index=False)
    json_write(PATHS["derived"] / "abstention_split.json", out)
    LOG.log("abstention", **out["by_category"])
    return out


def omniscience_index(graded: pd.DataFrame) -> pd.DataFrame:
    """PLAN §8.2 — (n_correct - n_incorrect) / n_total * 100, abstain scores 0."""
    c = graded[(graded.variant == "C") & (graded.sample_idx == 0)]
    rows = []
    for (model, tier), g in c.groupby(["model", "tier"]):
        n = len(g)
        answered = g[g.decision == "ANSWER"]
        n_cor = int(answered["correct"].sum())
        n_inc = int((~answered["correct"].astype(bool)).sum())
        n_abs = int((g.decision == "PASS").sum())
        rows.append(dict(model=model, tier=tier, n=n, n_correct=n_cor, n_incorrect=n_inc,
                         n_abstain=n_abs,
                         omniscience_index=round((n_cor - n_inc) / n * 100, 2) if n else np.nan,
                         abstention_rate=round(n_abs / n, 4) if n else np.nan))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_parquet(PATHS["derived"] / "omniscience_index.parquet", index=False)
    return df


def test_h3_base_vs_instruct(signals: pd.DataFrame, abst: dict, cfg: Config) -> dict:
    """PLAN §13 H3 / Gate 4 — unwarranted-stated-confidence delta between the
    7B base and instruct variants, with the missed-knowledge guard."""
    a, b = "qwen2.5-7b-base", "qwen2.5-7b-instruct"
    test = signals[(signals.split == "test") & signals.model.isin([a, b])].copy()
    if test.model.nunique() < 2:
        r = {"available": False, "verdict": "H3 untested — both 7B variants required"}
        json_write(PATHS["derived"] / "h3_model_delta.json", r)
        return r
    thr = cfg.QUADRANT_THRESHOLD
    test["other_cal"] = test[["behavioral_cal", "internal_cal"]].mean(axis=1, skipna=True)

    # AUDIT finding 7 / the Figure 3 defect.
    #
    #     ((verbal_cal >= thr) & (other_cal < thr)).astype(float)
    #
    # NaN >= thr is False, so every row where the base model's verbal signal
    # failed to parse became a hard 0.0 — indistinguishable from a genuine
    # "not overconfident". `.dropna()` then had nothing to drop, and
    # hopeful_rate_base came out at exactly 0.0. The bootstrap CI that
    # "excluded zero" was measuring missing data, not a behavioural
    # difference. The indicator is now defined ONLY over rows where both
    # signals are present; everything else is NaN and is dropped explicitly.
    valid = np.isfinite(test["verbal_cal"].astype(float)) & np.isfinite(test["other_cal"].astype(float))
    test["hopeful"] = np.where(valid,
                               ((test["verbal_cal"] >= thr) & (test["other_cal"] < thr)).astype(float),
                               np.nan)
    n_excluded = int((~valid).sum())
    excluded_by_model = {m: int((~valid & (test["model"] == m)).sum()) for m in (a, b)}

    # Match on questions BOTH variants scored validly, not merely on questions
    # both variants were asked.
    ok = test[valid]
    shared = set(ok[ok.model == a].qid) & set(ok[ok.model == b].qid)
    matched = ok[ok.qid.isin(shared)]
    xa = matched[matched.model == a]["hopeful"].dropna().values
    xb = matched[matched.model == b]["hopeful"].dropna().values
    if xa.size < 20 or xb.size < 20:
        r = {"available": False, "n_matched": int(len(shared)),
             "n_excluded_missing_signal": n_excluded,
             "excluded_by_model": excluded_by_model,
             "verdict": (f"H3 untested — only {xa.size}/{xb.size} matched rows have BOTH a verbal "
                         f"and a behavioral/internal value. The pre-repair code would have "
                         f"reported a rate of 0.0 here rather than declining to test.")}
        json_write(PATHS["derived"] / "h3_model_delta.json", r)
        LOG.log("h3", available=False, n_a=int(xa.size), n_b=int(xb.size))
        return r

    delta = bootstrap_diff_ci(xa, xb, np.mean, cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_CI, cfg.SEED)
    mk = {m: np.mean([v["missed_knowledge_rate"] for k, v in abst.get("per_cell", {}).items()
                      if k.startswith(m)] or [np.nan]) for m in (a, b)}
    mk_rise = float(mk[b] - mk[a]) if all(np.isfinite(list(mk.values()))) else float("nan")
    guard_ok = bool(not np.isfinite(mk_rise) or mk_rise < delta["delta"])

    # An elevated rate is not the same as "confident and wrong". Report the
    # correctness of the flagged rows and the confident-incorrect rate
    # separately, so the two are never conflated in the write-up.
    def rate(frame, col):
        v = frame[col].astype(float).dropna()
        return float(v.mean()) if len(v) else float("nan")
    per_model = {}
    for m in (a, b):
        g = matched[matched.model == m]
        flagged = g[g["hopeful"] == 1.0]
        per_model[m] = {
            "n": int(len(g)),
            "unwarranted_stated_confidence_rate": rate(g, "hopeful"),
            "flagged_rows_correct_rate": rate(flagged, "correct"),
            "confident_incorrect_rate": float(((g["verbal_cal"] >= thr) & (g["correct"] == 0)).mean())
                                         if len(g) else float("nan"),
            "overall_correct_rate": rate(g, "correct"),
        }

    out = {"available": True, "n_matched": int(len(shared)),
           "n_excluded_missing_signal": n_excluded,
           "excluded_by_model": excluded_by_model,
           "metric": "unwarranted stated confidence (Performative Certainty) rate",
           "unwarranted_rate_base": float(np.mean(xa)),
           "unwarranted_rate_instruct": float(np.mean(xb)),
           # legacy keys, deprecated — kept so older artefacts keep loading
           "hopeful_rate_base": float(np.mean(xa)),
           "hopeful_rate_instruct": float(np.mean(xb)),
           "per_model": per_model,
           "delta_base_minus_instruct": delta,
           "missed_knowledge_rate": {k: (None if not np.isfinite(v) else float(v)) for k, v in mk.items()},
           "missed_knowledge_rise": mk_rise, "guard_passes": guard_ok,
           "h3_pass": bool(delta["lo"] > 0 and guard_ok)}
    out["verdict"] = (
        "H3 descriptive result — in this evaluation setting instruction tuning shifts the rate of "
        "unwarranted stated confidence and the recoverability of missed knowledge under forced "
        "answering. Reported as a trade-off, not as evidence that one variant is better calibrated."
        if out["h3_pass"] else
        "H3 falsified / null — the base-minus-instruct CI includes 0, or the missed-knowledge "
        "guard fired.")
    json_write(PATHS["derived"] / "h3_model_delta.json", out)
    LOG.log("h3", pass_=out["h3_pass"], delta=round(delta["delta"], 4),
            excluded=n_excluded)
    return out


def gate3_filter_sweep(sweep: pd.DataFrame, gate3: "dict | None", cfg: Config
                        ) -> tuple[pd.DataFrame, list[str]]:
    """Drop cells that failed Gate 3 from a probe sweep.

    Gate 3 enforcement reached `internal_cal` but stopped there, so the depth
    curves — which the audit calls the best finding in the repo — still
    included cells whose p0 negative control had failed. A depth-of-onset
    result computed from activations the tap is known to have read wrongly is
    the original defect wearing a different figure.
    """
    if not len(sweep) or not cfg.GATE3_ENFORCE:
        return sweep, []
    if not gate3:
        # Fail CLOSED, matching `internal_scores`. Returning the unfiltered
        # sweep here meant one dropped keyword argument reverted the whole H4
        # repair silently — and an empty `gate3_excluded_cells` then reads as
        # "everything passed" rather than "enforcement never ran".
        raise RuntimeError(
            "gate3_filter_sweep: GATE3_ENFORCE is on but no Gate 3 verdict was supplied. "
            "Pass gate3=gate3_verdict(...), or set GATE3_ENFORCE=False deliberately.")
    ok = {c for c, v in gate3.items() if v.get("passes")}
    excluded = sorted(set(sweep["cell"]) - ok)
    return sweep[sweep["cell"].isin(ok)].copy(), excluded


def test_h4_depth(sweep: pd.DataFrame, cfg: Config, gate3: "dict | None" = None) -> dict:
    """PLAN §13 H4 — onset percentile (first percentile reaching AUROC >= gate)
    for retrieval vs reasoning, with a bootstrap CI on the difference."""
    sweep, excluded = gate3_filter_sweep(sweep, gate3, cfg)
    if not len(sweep):
        return {"available": False, "gate3_excluded_cells": excluded,
                "reason": ("every cell failed Gate 3" if excluded else "no probe sweep")}
    onsets = []
    for (model, tier), g in sweep.groupby(["model", "tier"]):
        g = g.sort_values("layer_pct")
        hit = g[g["auroc_cal"] >= cfg.AUROC_GATE]
        onsets.append(dict(model=model, tier=tier, family=TIER_SPECS[tier]["family"],
                           params_b=MODEL_SPECS[model]["params_b"],
                           onset=float(hit["layer_pct"].iloc[0]) if len(hit) else np.nan,
                           reached=bool(len(hit)),
                           asymptote=float(g["auroc_cal"].max())))
    odf = pd.DataFrame(onsets)
    odf.to_parquet(PATHS["derived"] / "h4_onsets.parquet", index=False)
    ret = odf[(odf.family == "retrieval") & odf.reached]["onset"].values
    rea = odf[(odf.family == "reasoning") & odf.reached]["onset"].values
    delta = bootstrap_diff_ci(rea, ret, np.mean, cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_CI, cfg.SEED) \
        if ret.size and rea.size else {"delta": np.nan, "lo": np.nan, "hi": np.nan, "excludes_zero": False}
    scale = {}
    if odf["reached"].any():
        r = odf[odf.reached]
        if r["params_b"].nunique() > 2:
            scale = {"spearman_onset_vs_scale": spearman_with_ci(
                r["params_b"].values, r["onset"].values, cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_CI, cfg.SEED)}
    out = {"available": True, "onsets": odf.to_dict("records"),
           "mean_onset_retrieval": float(ret.mean()) if ret.size else None,
           "mean_onset_reasoning": float(rea.mean()) if rea.size else None,
           "delta_reasoning_minus_retrieval": delta,
           "n_reasoning_never_reached": int((~odf[odf.family == "reasoning"]["reached"]).sum()),
           **scale,
           "h4_pass": bool(delta["excludes_zero"] and (delta["lo"] > 0))}
    out["verdict"] = ("H4 supported — internal signal onsets later for reasoning than retrieval"
                      if out["h4_pass"] else
                      "H4 falsified / null — onset curves overlap, or reasoning never reaches the gate")
    out["gate3_excluded_cells"] = excluded
    out["n_cells"] = int(sweep["cell"].nunique())
    if excluded:
        out["coverage_note"] = (
            f"{len(excluded)} cell(s) excluded for failing Gate 3 — the depth curves are "
            f"computed over {out['n_cells']} cells, not the full committed grid: {excluded}")
    json_write(PATHS["derived"] / "h4_depth.json", out)
    LOG.log("h4", pass_=out["h4_pass"], ret=out["mean_onset_retrieval"],
            rea=out["mean_onset_reasoning"], gate3_excluded=len(excluded))
    return out


def hierarchical_regression(signals: pd.DataFrame, cfg: Config,
                             bank: "dict | None" = None) -> dict:
    """PLAN §8·4 — ONE pooled multi-level model across the grid, not 30
    per-cell fits.

    AUDIT finding 6 / improvement 10: the previous fit carried a random
    intercept for QUESTION only. With nothing absorbing model capability or
    tier difficulty, every cell-level baseline was pushed into the fixed
    effects, so a coefficient on `is_long` could be reading "this tier is
    hard" rather than "this question is long". Random intercepts now cover
    model, tier and the model x tier cell as well, and the question-level
    characteristics enter as fixed effects so their estimates are read WITHIN
    cell baselines.

    Both cell-level and pooled estimates are reported: `per_cell` gives the
    unpooled fit for comparison, so the shrinkage the multi-level model
    applies is visible rather than implicit.
    """
    import statsmodels.formula.api as smf

    d = signals[signals.split == "test"].copy()
    n_test = len(d)
    verbal_empty = bool(n_test and d["verbal_cal"].isna().all())
    d = d.dropna(subset=["verbal_cal", "correct"])
    if verbal_empty:
        return {"available": False, "n": 0, "n_test_rows": n_test,
                "reason": "verbal_cal is entirely NaN on the test split — every cell was "
                          "excluded by the §8·6 degeneracy pre-flight, so there is no verbal "
                          "axis to regress on (this is not 'insufficient rows')"}
    # Missingness here is NOT at random. `confidence_behavioral` is NaN exactly
    # when fewer than ENTROPY_MIN_VALID samples parsed — i.e. on the questions
    # the model could not answer legibly — and it concentrates in particular
    # cells. Filling those with the column mean hands the hardest questions an
    # average-difficulty value, biasing the behavioral coefficient toward zero
    # and understating its standard error. Mean-imputation is kept so the fit
    # has complete rows, but each imputed column now carries its own
    # missingness indicator, so the model can absorb the difference instead of
    # attributing it to the signal.
    imputed: dict[str, int] = {}
    all_missing: list[str] = []
    near_all_missing: dict[str, float] = {}
    for c in ("behavioral_cal", "internal_cal"):
        miss = d[c].isna()
        imputed[c] = int(miss.sum())
        frac = float(miss.mean()) if len(d) else 0.0
        if 0.98 <= frac < 1.0:
            # Not "all missing", so the branch below does not fire — but a
            # column imputed from one or two surviving rows is a near-constant
            # standing in for a signal, and its coefficient means nothing.
            near_all_missing[c] = frac
        if miss.all():
            # Wholly absent, not partly missing. `fillna(mean)` is a no-op when
            # the mean is itself NaN, and the missingness indicator is dropped
            # precisely here (it has one distinct value), so the all-NaN column
            # would survive into the formula and patsy would drop EVERY row.
            # This is reachable, not hypothetical: while the activation-tap
            # defect persists the p0 control fails grid-wide, every cell is
            # skipped, and `internal_cal` is 100% NaN.
            all_missing.append(c)
            continue
        if miss.any():
            d[c + "_missing"] = miss.astype(int)
        d[c] = d[c].fillna(d[c].mean())
    if len(d) < 50:
        return {"available": False, "n": int(len(d)), "reason": "insufficient test rows"}

    d["log_params"] = np.log(d["params_b"])
    # `layer_pct` is the winning probe depth — an internal-probe artefact. When
    # the internal signal is gone, `best_layer_pct` is gone with it, and
    # filling 50.0 makes `layer_pct` a constant, so `layer_pct:is_reasoning`
    # becomes a scalar multiple of `is_reasoning` and the design matrix is
    # singular. It has to leave alongside the signal it describes.
    d["layer_pct"] = d.get("best_layer_pct", pd.Series(np.nan, index=d.index))
    layer_pct_usable = bool(d["layer_pct"].notna().any() and d["layer_pct"].nunique() > 1)
    d["layer_pct"] = d["layer_pct"].fillna(50.0)
    d["is_reasoning"] = (d["family"] == "reasoning").astype(int)
    d["cell"] = d["model"].astype(str) + "__" + d["tier"].astype(str)
    if near_all_missing:
        LOG.log("hlr_signal_near_empty", signals=near_all_missing,
                note="imputed from almost no surviving rows — the coefficient is a "
                     "near-constant, not a measurement")
    if all_missing:
        LOG.log("hlr_signal_dropped", signals=all_missing,
                note="entirely NaN on the test split — dropped from the model rather than "
                     "left in the formula, where it would silently drop every row")

    # Question characteristics as fixed effects (PLAN §13 H2 features).
    feat_cols: list[str] = []
    if bank is not None:
        want = ["is_long", "has_number", "has_multi_entity", "has_year"]
        feats = (question_features(bank)[["tier", "qid"] + want]
                 .drop_duplicates(subset=["tier", "qid"], keep="first"))
        # Drop any same-named column already on `signals` first. With
        # suffixes=("", "_f") the incumbent wins and the question feature
        # arrives as `is_long_f`, which is never added to `fixed_effects` — the
        # model silently loses the covariate it was built to include.
        d = d.drop(columns=[c for c in want if c in d.columns])
        n_before = len(d)
        d = d.merge(feats, on=["tier", "qid"], how="left", validate="m:1")
        if len(d) != n_before:
            raise RuntimeError(
                f"question_features merge changed the row count {n_before} -> {len(d)}; "
                f"the question bank has duplicate (tier, qid) keys")
        for c in want:
            if c in d:
                d[c] = d[c].fillna(False).astype(int)
                if d[c].nunique() > 1:
                    feat_cols.append(c)
    # `in_band` is the pilot-accuracy base-rate covariate that replaced cell
    # deletion (AUDIT finding 9): keep the control, drop the deletion. It is
    # None for a cell with no pilot record, and a partly-unknown covariate is
    # left out rather than coerced — `astype(int)` on None does not fail
    # loudly, it fails usefully-looking.
    if "in_band" in d and d["in_band"].notna().all() and d["in_band"].nunique() > 1:
        d["in_band"] = d["in_band"].astype(bool).astype(int)
        feat_cols.append("in_band")
    elif "in_band" in d and not d["in_band"].notna().all():
        out_note = int(d["in_band"].isna().sum())
        LOG.log("hlr_in_band_dropped", unknown_rows=out_note,
                note="in_band omitted from the model: some cells have no pilot record")

    miss_cols = [c + "_missing" for c in ("behavioral_cal", "internal_cal")
                 if c + "_missing" in d and d[c + "_missing"].nunique() > 1]
    signal_cols = [c for c in ("verbal_cal", "behavioral_cal", "internal_cal")
                   if c not in all_missing]

    def drop_degenerate(cols: list[str]) -> tuple[list[str], dict]:
        """Remove constant columns and exact duplicates of an earlier column.

        A constant is collinear with the intercept and an exact duplicate is
        collinear with its twin; either makes the design matrix singular and
        the whole fit returns `method: "failed"`. Two of the question features
        coincide whenever a tier's only numeric content IS a year, which is
        common in the retrieval tiers — so this is a real configuration, not a
        pathological one.
        """
        kept, dropped = [], {}
        for c in cols:
            col = d[c]
            if col.nunique(dropna=False) <= 1:
                dropped[c] = "constant"
                continue
            twin = next((k for k in kept if d[k].equals(col)), None)
            if twin is not None:
                dropped[c] = f"identical to {twin}"
                continue
            kept.append(c)
        return kept, dropped

    covariates, dropped_degenerate = drop_degenerate(
        ["is_reasoning", "log_params"] + feat_cols + miss_cols)
    if dropped_degenerate:
        LOG.log("hlr_degenerate_terms", dropped=dropped_degenerate,
                note="collinear with the intercept or with another term; the fit would be "
                     "singular with them in")
    fixed = signal_cols + covariates
    def _usable(term: str) -> bool:
        for part in term.split(":"):
            if part in ("internal_cal",) and part not in signal_cols:
                return False
            if part == "layer_pct" and not layer_pct_usable:
                return False
            if part in ("is_reasoning", "log_params") and part not in covariates:
                return False
        return True
    interactions = [t for t in ("is_reasoning:internal_cal", "log_params:internal_cal",
                                "layer_pct:is_reasoning") if _usable(t)]
    formula = "correct ~ " + " + ".join(fixed + interactions)
    out: dict[str, Any] = {"available": True, "n": int(len(d)), "formula": formula,
                           "fixed_effects": fixed,
                           "imputed_counts": imputed,
                           "missingness_indicators": miss_cols,
                           "signals_dropped_all_missing": all_missing,
                           "signals_near_all_missing": near_all_missing,
                           "terms_dropped_degenerate": dropped_degenerate,
                           # What was ATTEMPTED. `random_intercepts_fitted` —
                           # written only on a converged VB fit — is what was
                           # actually estimated. Reporting the wish list
                           # unconditionally meant a cluster-robust fallback,
                           # which has no random effects at all, still claimed
                           # four of them.
                           "random_intercepts_requested": ["question", "model", "tier",
                                                           "model_x_tier"],
                           "random_intercepts_fitted": [],
                           "n_questions": int(d["qid"].nunique()),
                           "n_cells": int(d["cell"].nunique())}

    # ---- unpooled per-cell fits, for the shrinkage comparison -----------
    per_cell: dict[str, dict] = {}
    # The same surviving signals as the pooled fit. Hardcoding all three meant
    # every per-cell fit died with the identical zero-size / singular error the
    # pooled path was repaired for.
    cell_formula = "correct ~ " + " + ".join(
        signal_cols + [c for c in feat_cols if c in covariates])
    for cell, g in d.groupby("cell"):
        if len(g) < 40 or g["correct"].nunique() < 2:
            continue
        try:
            m = smf.logit(cell_formula, data=g).fit(disp=0)
            per_cell[cell] = {"n": int(len(g)),
                              "params": {k: float(v) for k, v in m.params.items()},
                              "pvalues": {k: float(v) for k, v in m.pvalues.items()}}
        except Exception as exc:                          # noqa: BLE001
            per_cell[cell] = {"n": int(len(g)), "error": str(exc)[:120]}
    out["per_cell"] = per_cell

    # ---- pooled multi-level fit ----------------------------------------
    method = cfg.HLR_METHOD
    if method in ("auto", "bayes_mixed"):
        # Random intercepts at every level the design nests: question, model,
        # tier, and the model x tier cell. `vcp` groups are variance
        # components, all fitted simultaneously in one model.
        vc = {"question": "0 + C(qid)", "model": "0 + C(model)",
              "tier": "0 + C(tier)", "model_x_tier": "0 + C(cell)"}
        for drop in ([], ["model_x_tier"], ["model_x_tier", "tier"], ["model_x_tier", "tier", "model"]):
            use = {k: v for k, v in vc.items() if k not in drop}
            try:
                from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
                m = BinomialBayesMixedGLM.from_formula(formula, use, d)
                # Non-convergence is a WARNING, not an exception. Without this
                # the ladder never degraded: the first rung "succeeded" every
                # time and the report said four random intercepts were fitted
                # when the optimiser had given up.
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    r = m.fit_vb(verbose=False)
                notes = [str(w.message) for w in caught]
                if any("converge" in n.lower() for n in notes):
                    raise RuntimeError("VB fit did not converge: " + notes[0][:120])
                out.update(method="bayes_mixed_glm",
                           converged=True, fit_warnings=notes[:5],
                           random_intercepts_fitted=sorted(use),
                           random_intercepts_dropped=sorted(drop),
                           params=dict(zip(r.model.exog_names, [float(x) for x in r.fe_mean])),
                           sd=dict(zip(r.model.exog_names, [float(x) for x in r.fe_sd])),
                           vc_sd=dict(zip(getattr(r.model, "vcp_names", []),
                                          [float(x) for x in getattr(r, "vcp_mean", [])])))
                json_write(PATHS["derived"] / "hierarchical_regression.json", out)
                LOG.log("hlr", method=out["method"], n=out["n"],
                        random=out["random_intercepts_fitted"])
                return out
            except Exception as exc:                      # noqa: BLE001
                out.setdefault("bayes_errors", {})["+".join(sorted(use))] = str(exc)[:200]

    # ---- fallback: cluster-robust logit, clustered on the CELL ----------
    try:
        m = smf.logit(formula, data=d).fit(disp=0, cov_type="cluster",
                                           cov_kwds={"groups": d["cell"]})
        out.update(method="logit_cluster_robust_cell",
                   cluster_unit="model x tier cell",
                   params={k: float(v) for k, v in m.params.items()},
                   pvalues={k: float(v) for k, v in m.pvalues.items()},
                   conf_int={k: [float(a), float(b)] for k, (a, b) in m.conf_int().iterrows()},
                   pseudo_r2=float(m.prsquared), summary=str(m.summary()))
    except Exception as exc:                              # noqa: BLE001
        out.update(method="failed", error=str(exc)[:300])
    json_write(PATHS["derived"] / "hierarchical_regression.json", out)
    LOG.log("hlr", method=out.get("method"), n=out["n"])
    return out


def popularity_contrast(signals: pd.DataFrame, graded: pd.DataFrame, cfg: Config) -> dict:
    """R1 (top popularity quintile) vs R2 (bottom) vs R3 (adversarial).

    The popularity manipulation is the design's most distinctive feature and it
    was never analysed: the 25-80% band deleted every R2 and R3 cell, leaving
    the retrieval arm of H4 as a single tier — the easiest one (AUDIT findings
    5 and 9). With the band demoted to a covariate these cells run, so the
    gradient can finally be read.

    Per model, per tier: accuracy, the three calibrated signals, and the
    continuous verbal-minus-behavioral discrepancy. The contrast of interest is
    whether stated confidence tracks the popularity drop as closely as accuracy
    does — if verbal confidence stays flat while accuracy falls, that is
    unwarranted stated confidence measured against a manipulation rather than
    against a threshold.
    """
    POP = {"R1": "top_quintile", "R2": "bottom_quintile", "R3": "adversarial"}
    out: dict[str, Any] = {"tiers": POP, "available": False, "per_model": {}, "contrasts": {}}
    test = signals[(signals.split == "test") & signals.tier.isin(POP)]
    if not len(test):
        out["reason"] = "no retrieval-tier test rows — R1/R2/R3 produced no committed data"
        json_write(PATHS["derived"] / "popularity_gradient.json", out)
        return out

    def cell_stats(g: pd.DataFrame) -> dict:
        d = {"n": int(len(g)), "accuracy": float(g["correct"].mean())}
        for sig in SIGNALS:
            v = g[f"{sig}_cal"].astype(float).dropna()
            d[f"{sig}_cal_mean"] = float(v.mean()) if len(v) else None
        dd = g.get("delta_verbal_behavioral")
        dd = dd.astype(float).dropna() if dd is not None else pd.Series(dtype=float)
        d["delta_verbal_behavioral_mean"] = float(dd.mean()) if len(dd) else None
        return d

    for model, gm in test.groupby("model"):
        per_tier = {t: cell_stats(g) for t, g in gm.groupby("tier")}
        out["per_model"][model] = per_tier
        # R1 -> R2 is the clean popularity manipulation: same dataset, same
        # answer form, only the popularity quintile differs.
        if "R1" in per_tier and "R2" in per_tier:
            a, b = per_tier["R1"], per_tier["R2"]
            drop_acc = a["accuracy"] - b["accuracy"]
            dv = ((a.get("verbal_cal_mean") or np.nan) - (b.get("verbal_cal_mean") or np.nan))
            xa = gm[gm.tier == "R1"]["correct"].values.astype(float)
            xb = gm[gm.tier == "R2"]["correct"].values.astype(float)
            out["contrasts"][model] = {
                "accuracy_drop_R1_minus_R2": float(drop_acc),
                "verbal_cal_drop_R1_minus_R2": (None if not np.isfinite(dv) else float(dv)),
                "confidence_tracks_accuracy": (None if not np.isfinite(dv)
                                               else bool(dv >= 0.5 * drop_acc)),
                "accuracy_drop_ci": bootstrap_diff_ci(xa, xb, np.mean, cfg.N_BOOTSTRAP,
                                                      cfg.BOOTSTRAP_CI, cfg.SEED)
                                    if xa.size and xb.size else None,
            }
    out["available"] = bool(out["contrasts"])
    out["interpretation"] = (
        "confidence_tracks_accuracy=False means calibrated verbalized confidence fell by less "
        "than half as much as accuracy did when popularity dropped — stated confidence is not "
        "tracking the difficulty manipulation.")
    covered = sorted(set(test["tier"]))
    out["tiers_covered"] = covered
    if len(covered) < 3:
        out["coverage_warning"] = (f"only {covered} present; the popularity gradient needs R1 and "
                                   f"R2, and the adversarial contrast needs R3")
    json_write(PATHS["derived"] / "popularity_gradient.json", out)
    LOG.log("popularity_contrast", models=len(out["per_model"]), tiers=covered)
    return out


def difficulty_matched_view(signals: pd.DataFrame, cfg: Config, n_bins: int = 4) -> dict:
    """Within-cell, difficulty-matched signal comparison.

    The replacement for cell deletion (AUDIT finding 9, improvement 6). Items
    are binned by difficulty ESTIMATED ACROSS MODELS within a tier — the share
    of models that answered that question correctly — so the bin is a property
    of the question, not of the model being scored. Comparing signals inside a
    bin holds item difficulty fixed without discarding a single cell.
    """
    out: dict[str, Any] = {"available": False, "n_bins": n_bins, "per_tier": {}}
    test = signals[signals.split == "test"].copy()
    if not len(test):
        json_write(PATHS["derived"] / "difficulty_matched.json", out)
        return out

    # Cross-model difficulty of each question, within its tier.
    diff = (test.groupby(["tier", "qid"])["correct"].mean()
            .rename("item_difficulty").reset_index())
    diff["n_models"] = (test.groupby(["tier", "qid"])["model"].nunique().values)
    test = test.merge(diff, on=["tier", "qid"], how="left")

    degenerate: dict[str, dict] = {}
    for tier, gt in test.groupby("tier"):
        d = gt["item_difficulty"].astype(float)
        if d.nunique() < 2:
            # Every question in the tier has the same cross-model correctness
            # rate — usually because the tier is near 0% or near 100% for all
            # models. There is no within-tier difficulty variation to match on.
            degenerate[tier] = {"reason": "no cross-model difficulty variation",
                                "n_rows": int(len(gt)), "value": float(d.iloc[0]) if len(d) else None}
            continue
        try:
            gt = gt.assign(diff_bin=pd.qcut(d, q=min(n_bins, d.nunique()),
                                            labels=False, duplicates="drop"))
        except ValueError:
            continue
        per_bin = {}
        for b, gb in gt.groupby("diff_bin"):
            if len(gb) < 30:
                continue
            entry = {"n": int(len(gb)), "mean_item_difficulty": float(gb["item_difficulty"].mean()),
                     "base_rate": float(gb["correct"].mean()), "signals": {}}
            for sig in SIGNALS:
                v = gb[f"{sig}_cal"].astype(float)
                y = gb["correct"].astype(float)
                m = np.isfinite(v) & np.isfinite(y)
                if m.sum() < 20 or y[m].nunique() < 2:
                    continue
                entry["signals"][sig] = {
                    "n": int(m.sum()),
                    "auroc": safe_auroc(y[m].values.astype(int), v[m].values),
                    "ece": ece(v[m].values, y[m].values, cfg.ECE_BINS),
                    "brier": brier(v[m].values, y[m].values),
                    # Resolution is the term that survives base-rate matching:
                    # inside a bin the base rate is fixed, so this is signal,
                    # not difficulty.
                    "resolution": murphy_decomposition(v[m].values, y[m].values,
                                                       cfg.MURPHY_BINS).get("resolution"),
                }
            per_bin[str(int(b))] = entry
        if per_bin:
            out["per_tier"][tier] = {"bins": per_bin,
                                     "n_models": int(gt["model"].nunique()),
                                     "n_questions": int(gt["qid"].nunique())}
    out["available"] = bool(out["per_tier"])
    out["degenerate_tiers"] = degenerate
    out["note"] = ("Difficulty is the cross-model correctness rate of the question within its "
                   "tier, so bins are a property of the item. Comparing signals within a bin "
                   "controls the base rate WITHOUT deleting any cell.")
    if degenerate:
        out["limitation"] = (
            "Matching needs within-tier difficulty variation, and a tier where every model "
            "scores near 0% (or near 100%) has none — which is exactly the regime the 25-80% "
            "band used to delete. For those tiers this control is unavailable, and the "
            "base-rate adjustment rests entirely on the GLMM's model x tier random intercept. "
            "Say so rather than reporting a matched comparison that did not happen: "
            f"{sorted(degenerate)}.")
        LOG.log("difficulty_matched_degenerate", tiers=sorted(degenerate),
                note="no within-tier difficulty variation; matching unavailable for these")
    json_write(PATHS["derived"] / "difficulty_matched.json", out)
    LOG.log("difficulty_matched", tiers=len(out["per_tier"]))
    return out


def correlation_table(signals: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """PLAN §8·7 — Spearman on RAW, Pearson on CALIBRATED (both reported)."""
    rows = []
    test = signals[signals.split == "test"]
    for (model, tier), g in list(test.groupby(["model", "tier"])) + [(("POOLED", "ALL"), test)]:
        for a, b in (("verbal", "behavioral"), ("verbal", "internal"), ("behavioral", "internal")):
            raw = g[[f"{a}_raw", f"{b}_raw"]].dropna()
            cal = g[[f"{a}_cal", f"{b}_cal"]].dropna()
            sp = spearman_with_ci(raw[f"{a}_raw"].values, raw[f"{b}_raw"].values,
                                  min(cfg.N_BOOTSTRAP, 500), cfg.BOOTSTRAP_CI, cfg.SEED) if len(raw) > 5 else {}
            pe = (float(sps.pearsonr(cal[f"{a}_cal"], cal[f"{b}_cal"]).statistic)
                  if len(cal) > 5 and cal[f"{a}_cal"].std() > 0 and cal[f"{b}_cal"].std() > 0 else np.nan)
            rows.append(dict(model=model, tier=tier, pair=f"{a}-{b}", n_raw=len(raw), n_cal=len(cal),
                             spearman_raw=sp.get("rho", np.nan), spearman_lo=sp.get("lo", np.nan),
                             spearman_hi=sp.get("hi", np.nan), pearson_calibrated=pe))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_parquet(PATHS["derived"] / "correlations.parquet", index=False)
    return df
# %%
# ============================================================================
# CELL 18 — post-hoc LLM judge (optional AUDIT layer, runs last)
#
# The judge is deliberately NOT the primary label. Primary grading stays
# deterministic (cell 8); the judge runs after every generation model has been
# freed, reads only the saved JSON, and produces a SECOND opinion whose
# agreement with the deterministic grader is the reported Gate 1 statistic
# (PLAN §7 "report the labeling method and manual-agreement rate", §16 Gate 1).
#
# Quantization is fine here and does NOT violate PLAN §9.1: that rule protects
# the hidden states the probe reads, and the judge is never probed.
# ============================================================================


@dataclass
class JudgeConfig:
    ENABLED: bool = False                     # flip on for the audit pass
    # Default ladder is chosen to load NATIVELY in bf16, because a quantized
    # checkpoint needs its kernel package installed (autoawq / gptqmodel /
    # bitsandbytes) and those often lack kernels for the newest GPUs. 32B in
    # bf16 is ~65 GB and fits a 96 GB card outright; swap to an AWQ id when
    # the kernels are available and VRAM is tighter.
    MODEL: str = "Qwen/Qwen2.5-32B-Instruct"
    FALLBACKS: tuple[str, ...] = ("Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
    LOAD: str = "auto"                        # auto | awq | gptq | bnb4 | native
    MAX_NEW_TOKENS: int = 12
    BATCH_SIZE: int = 16
    # What to send to the judge:
    #   unresolved  - items no deterministic tier could decide
    #   fuzzy       - anything resolved by nli / string_fallback
    #   audit       - a random sample of deterministically-graded items (Gate 1)
    #   all         - every graded row (expensive)
    TARGETS: tuple[str, ...] = ("unresolved", "fuzzy", "audit")
    AUDIT_SAMPLE_PER_FAMILY: int = 100        # Gate 1 sample size per grader family
    FREE_AFTER: bool = True


JUDGE = JudgeConfig(ENABLED=False)

JUDGE_PROMPT = (
    "You are grading one short answer against a reference answer. "
    "Judge only semantic equivalence: is the given answer the same fact/value as the reference? "
    "Ignore wording, formatting, capitalisation and extra words.\n\n"
    "Reply in EXACTLY this format and nothing else:\n"
    "VERDICT: <CORRECT or INCORRECT>\n\n"
    "Question: {question}\n"
    "Reference answer: {gold}\n"
    "Given answer: {pred}\n"
)


def load_judge(jc: JudgeConfig, cfg: Config):
    from transformers import AutoModelForCausalLM as _AM, AutoTokenizer as _AT
    last_err = None
    for mid in (jc.MODEL, *jc.FALLBACKS):
        try:
            kwargs: dict[str, Any] = dict(device_map="auto", low_cpu_mem_usage=True,
                                          attn_implementation=cfg.ATTN_IMPL)
            mode = jc.LOAD
            if mode == "auto":
                # Resolve to something the environment can actually load: a
                # pre-quantized checkpoint needs its kernel package, and bnb4
                # needs bitsandbytes. With none present, run native and let
                # VRAM decide (the fallback ladder handles "too big").
                if "awq" in mid.lower() and importlib.util.find_spec("awq"):
                    mode = "awq"
                elif "gptq" in mid.lower() and (importlib.util.find_spec("gptqmodel")
                                                or importlib.util.find_spec("auto_gptq")):
                    mode = "gptq"
                elif importlib.util.find_spec("bitsandbytes"):
                    mode = "bnb4"
                else:
                    mode = "native"
            if mode in ("awq", "gptq"):
                kwargs["torch_dtype"] = torch.float16
            elif mode == "bnb4":
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=cfg.resolved_dtype(), bnb_4bit_use_double_quant=True)
            else:
                kwargs["torch_dtype"] = cfg.resolved_dtype()
            tok = _AT.from_pretrained(mid, padding_side="left")
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = _AM.from_pretrained(mid, **kwargs).eval()
            LOG.log("judge_loaded", model=mid, mode=mode, vram=vram_report())
            return LoadedModel(f"judge:{mid}", mdl, tok, {"hf_id": mid, "chat": True,
                                                          "params_b": 0.0, "layers": 0, "hidden": 0})
        except Exception as exc:                          # noqa: BLE001
            last_err = f"{mid}: {type(exc).__name__}: {exc}"
            LOG.log("judge_load_failed", model=mid, error=str(exc)[:160])
            free_cuda()
    raise RuntimeError(f"no judge model could be loaded. last error: {last_err}")


def judge_targets(graded: pd.DataFrame, jc: JudgeConfig, cfg: Config) -> pd.DataFrame:
    g = graded[graded.answer.notna()].copy()
    parts = []
    if "all" in jc.TARGETS:
        parts.append(g)
    else:
        if "unresolved" in jc.TARGETS:
            parts.append(g[~g.resolved.astype(bool)])
        if "fuzzy" in jc.TARGETS:
            parts.append(g[g.grader.isin(["nli", "string_fallback", "latex_normalized"])])
        if "audit" in jc.TARGETS:
            det = g[g.grader.isin(["alias_exact", "numeric", "symbolic", "no_answer"])]
            rng = np.random.default_rng(cfg.SEED)
            for _, fg in det.groupby("answer_form"):
                take = min(jc.AUDIT_SAMPLE_PER_FAMILY, len(fg))
                parts.append(fg.iloc[rng.choice(len(fg), take, replace=False)])
    if not parts:
        return g.head(0)
    out = pd.concat(parts).drop_duplicates(subset=["model", "tier", "qid", "variant", "sample_idx"])
    return out


@torch.no_grad()
def stage_judge(bank: dict, graded: pd.DataFrame, jc: JudgeConfig, cfg: Config) -> dict:
    """Runs before the hypothesis tests (PLAN §16 conditions H2 on Gate 1). Idempotent and resumable like every stage."""
    if not jc.ENABLED:
        return {"enabled": False}
    targets = judge_targets(graded, jc, cfg)
    ck = Checkpoint(PATHS["derived"] / "judge.jsonl",
                    ("model", "tier", "qid", "variant", "sample_idx"), cfg.CHECKPOINT_EVERY, cfg.RESUME)
    todo = [r for r in targets.itertuples()
            if not ck.has(model=r.model, tier=r.tier, qid=r.qid, variant=r.variant, sample_idx=r.sample_idx)]
    LOG.log("judge_start", targets=len(targets), todo=len(todo), model=jc.MODEL)

    if todo:
        qtext = {r["qid"]: r["question"] for t in bank for r in bank[t]}
        jm = load_judge(jc, cfg)
        try:
            for i in tqdm(range(0, len(todo), jc.BATCH_SIZE), desc="judge", leave=False):
                chunk = todo[i : i + jc.BATCH_SIZE]
                prompts = []
                for r in chunk:
                    msgs = [{"role": "user", "content": JUDGE_PROMPT.format(
                        question=qtext.get(r.qid, ""), gold=r.gold, pred=r.answer)}]
                    prompts.append(jm.tokenizer.apply_chat_template(msgs, tokenize=False,
                                                                    add_generation_prompt=True))
                enc = jm.tokenizer(prompts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=1024).to(jm.hidden_device)
                out = jm.model.generate(**enc, max_new_tokens=jc.MAX_NEW_TOKENS, do_sample=False,
                                        pad_token_id=jm.tokenizer.pad_token_id)
                texts = jm.tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for r, t in zip(chunk, texts):
                    v = (grab(t, "VERDICT") or t or "").upper()
                    verdict = True if "CORRECT" in v and "INCORRECT" not in v else (
                        False if "INCORRECT" in v else None)
                    ck.add({"model": r.model, "tier": r.tier, "qid": r.qid, "variant": r.variant,
                            "sample_idx": r.sample_idx, "grader": r.grader,
                            "deterministic_correct": bool(r.correct), "judge_correct": verdict,
                            "judge_raw": t.strip()[:200], "judge_model": jm.spec["hf_id"],
                            "answer_form": r.answer_form})
                if cfg.EMPTY_CACHE_EVERY_BATCHES and (i // max(jc.BATCH_SIZE, 1)) % cfg.EMPTY_CACHE_EVERY_BATCHES == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        finally:
            ck.flush()
            if jc.FREE_AFTER:
                free_model(jm, cfg, purge=False)

    jdf = pd.DataFrame(jsonl_read(PATHS["derived"] / "judge.jsonl"))
    if not len(jdf):
        return {"enabled": True, "n": 0}
    jdf.to_parquet(PATHS["derived"] / "judge.parquet", index=False)
    dec = jdf[jdf.judge_correct.notna()]
    agree_overall = float((dec.judge_correct == dec.deterministic_correct).mean()) if len(dec) else np.nan
    by_family = {}
    for form, g in dec.groupby("answer_form"):
        by_family[str(form)] = {"n": int(len(g)),
                                "agreement": round(float((g.judge_correct == g.deterministic_correct).mean()), 4)}
    by_grader = {str(k): {"n": int(len(g)),
                          "agreement": round(float((g.judge_correct == g.deterministic_correct).mean()), 4)}
                 for k, g in dec.groupby("grader")}
    res = {"enabled": True, "n": int(len(jdf)), "n_decided": int(len(dec)),
           "judge_model": str(jdf["judge_model"].iloc[0]),
           "agreement_overall": round(agree_overall, 4) if np.isfinite(agree_overall) else None,
           "agreement_by_answer_form": by_family, "agreement_by_grader": by_grader,
           "gate1_threshold": cfg.GATE1_AGREEMENT,
           "gate1_pass": bool(np.isfinite(agree_overall) and agree_overall >= cfg.GATE1_AGREEMENT),
           "note": ("Judge is a secondary audit label. Deterministic grading remains primary; "
                    "this agreement rate is the reported grading-sanity statistic (PLAN §7, §16 Gate 1). "
                    "Manual verification of a subsample is still required to close Gate 1 fully.")}
    json_write(PATHS["derived"] / "judge_agreement.json", res)
    LOG.log("judge_done", n=res["n_decided"], agreement=res["agreement_overall"],
            gate1=res["gate1_pass"])
    return res


def export_manual_check_sheet(bank: dict, graded: pd.DataFrame, cfg: Config) -> Path:
    """Gate 1 needs HUMAN verification of 50–100 items per grader family
    (PLAN §3, §16). This writes the sheet to hand-label; the judge above is an
    additional automated opinion, not a replacement for it.

    A filled sheet is NEVER overwritten. Re-running the pipeline used to drop a
    fresh blank template on top of the human labels, which is the most likely
    reason `manual_correct` reads 0/200 in two of the three result trees while
    `POINTS(Debojeet).md` records Gate 1 as completed at 97.5% (AUDIT finding
    8). A new template is written beside the labelled sheet instead, so extra
    rows can still be graded without destroying the existing ones.
    """
    qtext = {r["qid"]: r["question"] for t in bank for r in bank[t]}
    rng = np.random.default_rng(cfg.SEED)
    rows = []
    for form, g in graded[graded.answer.notna()].groupby("answer_form"):
        take = min(cfg.N_MANUAL_CHECK, len(g))
        for r in g.iloc[rng.choice(len(g), take, replace=False)].itertuples():
            rows.append(dict(answer_form=form, model=r.model, tier=r.tier, qid=r.qid,
                             question=qtext.get(r.qid, ""), gold=r.gold, model_answer=r.answer,
                             automated_correct=bool(r.correct), grader=r.grader,
                             manual_correct="", disagreement_note=""))
    df = pd.DataFrame(rows)
    p = PATHS["tables"] / "gate1_manual_check_sheet.csv"
    if p.exists():
        try:
            existing = pd.read_csv(p)
            n_filled = int(existing["manual_correct"].notna().sum()) if "manual_correct" in existing else 0
        except Exception:                                # noqa: BLE001
            existing, n_filled = None, 0
        if n_filled > 0:
            alt = PATHS["tables"] / "gate1_manual_check_sheet.NEW_TEMPLATE.csv"
            df.to_csv(alt, index=False)
            LOG.log("manual_sheet_preserved", path=str(p), filled_rows=n_filled,
                    template_written=str(alt),
                    note="human labels found — refusing to overwrite")
            return p
    df.to_csv(p, index=False)
    LOG.log("manual_sheet", path=str(p), rows=len(df))
    return p


def ingest_manual_sheet(path: Path, cfg: Config) -> dict:
    """Read a hand-labelled Gate 1 sheet back into the report.

    Accepts the usual spellings a human types into a spreadsheet. Rows left
    blank are UNGRADED and are excluded from the denominator — they are not
    counted as disagreements, and they are not counted as agreements either.
    """
    out = {"available": False, "path": str(path), "n_rows": 0, "n_labelled": 0,
           "agreement": None, "n_disagreements": 0, "disagreement_notes": [],
           "by_grader": {}}
    if not Path(path).exists():
        return out
    try:
        df = pd.read_csv(path)
    except Exception as exc:                             # noqa: BLE001
        out["error"] = str(exc)[:200]
        return out
    out["n_rows"] = int(len(df))
    if "manual_correct" not in df or "automated_correct" not in df:
        out["error"] = "sheet is missing manual_correct / automated_correct"
        return out

    TRUE = {"1", "1.0", "true", "t", "yes", "y", "correct"}
    FALSE = {"0", "0.0", "false", "f", "no", "n", "incorrect", "wrong"}

    def to_bool(v):
        s = str(v).strip().lower()
        if s in TRUE:
            return True
        if s in FALSE:
            return False
        return None

    df["_manual"] = df["manual_correct"].map(to_bool)
    df["_auto"] = df["automated_correct"].map(to_bool)
    lab = df[df["_manual"].notna() & df["_auto"].notna()]
    out["n_labelled"] = int(len(lab))
    if not len(lab):
        LOG.log("manual_sheet_ingest", labelled=0, note="sheet present but unlabelled")
        return out
    agree = (lab["_manual"] == lab["_auto"])
    out.update(available=True, agreement=float(agree.mean()),
               n_disagreements=int((~agree).sum()),
               passes=bool(agree.mean() >= cfg.GATE1_AGREEMENT),
               threshold=cfg.GATE1_AGREEMENT)
    if "grader" in lab:
        out["by_grader"] = {str(k): {"n": int(len(g)), "agreement": float((g["_manual"] == g["_auto"]).mean())}
                            for k, g in lab.groupby("grader")}
    if "disagreement_note" in lab:
        out["disagreement_notes"] = [str(x) for x in lab.loc[~agree, "disagreement_note"].dropna()
                                     if str(x).strip()][:20]
    LOG.log("manual_sheet_ingest", labelled=out["n_labelled"], agreement=round(out["agreement"], 4),
            passes=out["passes"])
    return out


def combined_gate1(judge: dict, manual: dict, cfg: Config) -> "bool | None":
    """Gate 1 verdict from the human sheet and the automated judge.

    The human sheet is authoritative when it exists — PLAN §16 asks for human
    verification and the judge is explicitly "an additional automated opinion,
    not a replacement for it". When neither source has data the verdict is
    None, which is NOT the same as False and is NOT the same as True: H2's
    conjunct treats an unfilled Gate 1 as a failing conjunct rather than
    quietly assuming it held.
    """
    if manual.get("available"):
        return bool(manual["passes"])
    jp = judge.get("gate1_pass")
    return None if jp is None else bool(jp)


# %%
# ============================================================================
# CELL 19 — figures (PLAN §15, Figures 1–4 + supplementary)
# Palette is a validated CVD-safe categorical order; hues are assigned in fixed
# slot order and never cycled. One y-axis per panel, always.
# ============================================================================
import matplotlib as mpl                                        # noqa: E402
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.colors import LinearSegmentedColormap           # noqa: E402
from matplotlib.lines import Line2D                             # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#256abf", "#6da7ec", "#f0efec", "#e87ba4", "#e34948", "#8f1f1f"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e6e5e1", "#ffffff"
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list("div_br", DIVERGING)

# Fixed slot per tier so a filtered chart never repaints the survivors.
TIER_COLOR = {t: PALETTE[i] for i, t in enumerate(TIER_SPECS)}
SIGNAL_COLOR = {"verbal": PALETTE[0], "behavioral": PALETTE[1], "internal": PALETTE[2]}
QUADRANT_COLOR = {"hopeful": PALETTE[0], "suppressed": PALETTE[1],
                  "agree_high": "#334155", "agree_low": "#94a3b8"}


def apply_style(cfg: Config) -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "legend.fontsize": 7.8, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK3, "ytick.color": INK3,
        "grid.color": GRID, "grid.linewidth": 0.7, "axes.grid": True, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 5.5,
        "figure.dpi": 110, "savefig.dpi": cfg.FIG_DPI, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save_fig(fig, name: str, cfg: Config, caption: str = "") -> list[Path]:
    paths = []
    for ext in cfg.FIG_FORMATS:
        p = PATHS["figures"] / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    if caption:
        (PATHS["figures"] / f"{name}.caption.txt").write_text(caption)
    plt.close(fig)
    LOG.log("figure_saved", name=name, files=len(paths))
    return paths


def reliability_curve(p: np.ndarray, y: np.ndarray, bins: int, min_n: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() < min_n:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def fig1_calibration(signals: pd.DataFrame, h1: dict, cfg: Config) -> None:
    """Figure 1 — calibrated P(correct) vs stated confidence, one curve per
    signal, with bootstrap bands. `predicted under H1`: verbal sits above the
    diagonal; the null is all three ON the diagonal with overlapping bands."""
    apply_style(cfg)
    test = signals[signals.split == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(cfg.FIG_WIDTH, cfg.FIG_WIDTH * 0.42))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.0, color=INK3, zorder=1, label="perfect calibration")
    rng = np.random.default_rng(cfg.SEED)
    for sig in SIGNALS:
        d = test[[f"{sig}_cal", "correct"]].dropna()
        if len(d) < 20:
            continue
        p, y = d[f"{sig}_cal"].values, d["correct"].values.astype(float)
        xs, ys, ns = reliability_curve(p, y, cfg.ECE_BINS)
        if not len(xs):
            continue
        boots = []
        for _ in range(300):
            i = rng.integers(0, len(p), len(p))
            bx, by, _ = reliability_curve(p[i], y[i], cfg.ECE_BINS)
            if len(bx) == len(xs):
                boots.append(by)
        c = SIGNAL_COLOR[sig]
        if boots:
            B = np.vstack(boots)
            ax.fill_between(xs, np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0),
                            color=c, alpha=0.16, lw=0, zorder=2)
        ax.plot(xs, ys, color=c, marker="o", mec=SURFACE, mew=1.2, zorder=3,
                label=f"{sig} (ECE {h1.get('pooled', {}).get(sig, {}).get('ece', float('nan')):.3f})")
    ax.set(xlabel="stated / predicted confidence", ylabel="observed P(correct)",
           xlim=(0, 1), ylim=(0, 1), title="Calibration by signal (test split)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=7.2,
              borderpad=0.3, labelspacing=0.25, handlelength=1.4, framealpha=0.92)

    ax2 = axes[1]
    comps, labels = ["reliability", "resolution"], []
    width = 0.36
    max_val = 0.0
    for k, sig in enumerate(SIGNALS):
        v = h1.get("pooled", {}).get(sig, {})
        if "reliability" not in v:
            continue
        labels.append(sig)
        for j, comp in enumerate(comps):
            val = v.get(comp, np.nan)
            if np.isfinite(val):
                max_val = max(max_val, val)
            ax2.bar(len(labels) - 1 + (j - 0.5) * width, val, width * 0.92,
                    color=SIGNAL_COLOR[sig], alpha=1.0 if j == 0 else 0.45,
                    edgecolor=SURFACE, linewidth=1.2)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    if max_val > 0:
        ax2.set_ylim(0, max_val * 1.25)
    ax2.set(ylabel="Brier component", title="Murphy decomposition")
    ax2.legend(handles=[Line2D([], [], color=INK3, lw=6, alpha=1.0, label="reliability (lower better)"),
                        Line2D([], [], color=INK3, lw=6, alpha=0.45, label="resolution (higher better)")],
              loc="upper right", fontsize=7.2, borderpad=0.3, labelspacing=0.25, framealpha=0.92)
    fig.tight_layout()
    save_fig(fig, "fig1_calibration", cfg,
             "Figure 1 — calibration curves per signal, test split. Predicted under H1: the "
             "verbalized curve sits above the diagonal (runs hot) while behavioral and internal "
             "hug it. Null: all three on the diagonal with overlapping bands.")


def fig2_quadrant(cfg: Config, abst: dict) -> None:
    """Figure 2 — quadrant scatter + abstention split companion (PLAN §15)."""
    apply_style(cfg)
    path = PATHS["derived"] / "quadrants.parquet"
    if not path.exists():
        return
    q = pd.read_parquet(path)
    fig, axes = plt.subplots(1, 2, figsize=(cfg.FIG_WIDTH, cfg.FIG_WIDTH * 0.44),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    thr = cfg.QUADRANT_THRESHOLD
    ax.axvline(thr, color="#64748b", ls="--", lw=1.2, zorder=1)
    ax.axhline(thr, color="#64748b", ls="--", lw=1.2, zorder=1)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#64748b")
        ax.spines[spine].set_linewidth(1.0)
    for name in ("agree_low", "agree_high", "suppressed", "hopeful"):   # mismatches drawn on top
        s = q[q.quadrant == name]
        if not len(s):
            continue
        ax.scatter(s["verbal_cal"], s["other_cal"], s=16, color=QUADRANT_COLOR[name],
                   edgecolors=SURFACE, linewidths=0.4, alpha=0.85, zorder=3 if "agree" not in name else 2,
                   label=f"{quadrant_label(name)} (n={len(s)})")
    ax.set(xlabel="calibrated verbalized confidence", ylabel="calibrated behavioral / internal",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), title="Signal mismatch quadrants (test split)")
    ax.legend(loc="lower right", markerscale=1.3, fontsize=7.6, framealpha=0.92, edgecolor="#cbd5e1")

    ax2 = axes[1]
    cats = ["justified_hedge", "missed_knowledge", "unresolved"]
    vals = [abst.get("by_category", {}).get(c, 0) for c in cats]
    ax2.barh(range(len(cats)), vals, color=[PALETTE[2], PALETTE[1], "#cbd5e1"],
             edgecolor=SURFACE, linewidth=1.4, height=0.62)
    for i, v in enumerate(vals):
        ax2.text(v, i, f" {v}", va="center", ha="left", color=INK2, fontsize=7.8)
    ax2.set_yticks(range(len(cats)))
    ax2.set_yticklabels([c.replace("_", " ") for c in cats])
    ax2.set(xlabel="Format C passes", title="Abstention split (PLAN §4.1)")
    ax2.grid(axis="y", visible=False)
    for spine in ["bottom", "left"]:
        ax2.spines[spine].set_color("#64748b")
        ax2.spines[spine].set_linewidth(1.0)
    fig.tight_layout()
    save_fig(fig, "fig2_quadrant", cfg,
             "Figure 2 — epistemic-alignment quadrants: calibrated verbalized confidence vs "
             "calibrated behavioral/internal confidence, with the abstention split. "
             "Performative Certainty = stated confidence exceeds sample stability; Excessive "
             "Hedging = the reverse; Grounded Certainty and Honest Doubt are the congruent "
             "states. Predicted under H2: the two mismatch quadrants cluster by question type "
             "WITHIN cells. Null: uniform scatter. Note the axis threshold is a fixed 0.5 cut — "
             "see h2_quadrants.json `concentration` and `threshold_degenerate_cells` for how "
             "many cells sit entirely on one side of it.")


def fig3_model_delta(h3: dict, cfg: Config) -> None:
    """Figure 3 — base vs Instruct: unwarranted-stated-confidence rate and
    missed-knowledge rate.

    Drawn only when H3 is `available`. Before the repair, a base-model rate of
    0.0 produced entirely by NaN coercion was plotted here as a real bar
    (AUDIT finding 7); H3 now declines to report rather than plot missing data.
    """
    apply_style(cfg)
    if not h3.get("available"):
        return
    apply_style(cfg)
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH * 0.62, cfg.FIG_WIDTH * 0.42))
    groups = ["unwarranted stated\nconfidence", "missed knowledge"]
    base = [h3.get("unwarranted_rate_base", h3.get("hopeful_rate_base", np.nan)),
            (h3.get("missed_knowledge_rate", {}) or {}).get("qwen2.5-7b-base", np.nan)]
    inst = [h3.get("unwarranted_rate_instruct", h3.get("hopeful_rate_instruct", np.nan)),
            (h3.get("missed_knowledge_rate", {}) or {}).get("qwen2.5-7b-instruct", np.nan)]
    x = np.arange(len(groups))
    w = 0.34
    ax.bar(x - w / 2, base, w * 0.94, color=PALETTE[0], edgecolor=SURFACE, linewidth=1.4, label="7B base")
    ax.bar(x + w / 2, inst, w * 0.94, color=PALETTE[1], edgecolor=SURFACE, linewidth=1.4, label="7B Instruct")
    d = h3.get("delta_base_minus_instruct", {})
    if np.isfinite(d.get("lo", np.nan)):
        ax.errorbar(x[0], base[0], yerr=[[max(base[0] - (inst[0] + d["lo"]), 0)],
                                         [max((inst[0] + d["hi"]) - base[0], 0)]],
                    fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=4)
    for xi, v in zip(np.concatenate([x - w / 2, x + w / 2]), base + inst):
        if np.isfinite(v):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", color=INK2, fontsize=7.4)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set(ylabel="rate (test split)", title="Post-training delta (H3)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig3_model_delta", cfg,
             "Figure 3 — rate of unwarranted stated confidence (Performative Certainty) and "
             "missed-knowledge rate, Qwen2.5-7B base vs Instruct, over questions where BOTH a "
             "verbalized and a behavioral/internal value exist. Descriptive trade-off: "
             "instruction tuning shifts asserted-confidence disagreement and forced-answer "
             "recoverability together. Null: overlapping bars.")


def fig4_depth_curves(sweep: pd.DataFrame, cfg: Config) -> None:
    """Figure 4 — AUROC vs layer percentile, one line per tier, faceted by
    model size. The direct visual test of H4 (PLAN §6·7, §15)."""
    apply_style(cfg)
    if not len(sweep):
        return
    models = [m for m in cfg.active_models() if m in set(sweep["model"])]
    if not models:
        return
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(cfg.FIG_WIDTH, cfg.FIG_WIDTH * 0.34),
                             sharey=True, squeeze=False)
    for k, model in enumerate(models):
        ax = axes[0][k]
        g = sweep[sweep.model == model]
        ax.axhline(0.5, color=INK3, ls=(0, (4, 3)), lw=0.9, zorder=1)
        ax.axhline(cfg.AUROC_GATE, color=GRID, lw=1.0, zorder=1)
        for tier in cfg.active_tiers():
            t = g[g.tier == tier].sort_values("layer_pct")
            if not len(t):
                continue
            ax.plot(t["layer_pct"], t["auroc_cal"], color=TIER_COLOR[tier], marker="o",
                    mec=SURFACE, mew=1.0, zorder=3, label=tier)
            if t["auroc_null_p95"].notna().any():
                ax.fill_between(t["layer_pct"], 0.5, t["auroc_null_p95"],
                                color=GRID, alpha=0.55, lw=0, zorder=1)
        ax.set(xlabel="layer percentile", xticks=list(cfg.PERCENTILES), ylim=(0.35, 1.0),
               title=f"{model.replace('qwen2.5-', '').replace('-instruct', '')}"
                     f" ({MODEL_SPECS[model]['params_b']:.1f}B)")
        if k == 0:
            ax.set_ylabel("probe AUROC (calibration split)")
    handles = [Line2D([], [], color=TIER_COLOR[t], lw=2.2, marker="o", mec=SURFACE,
                      label=f"{t} · {TIER_SPECS[t]['family'][:4]}") for t in cfg.active_tiers()]
    handles.append(Line2D([], [], color=GRID, lw=6, label="label-shuffle null (p95)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7),
               bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    save_fig(fig, "fig4_depth_prediction", cfg,
             "Figure 4 — AUROC vs layer percentile, one line per tier, faceted by model size. "
             "Predicted under H4: retrieval tiers flat and high from ~0% depth; reasoning tiers "
             "at chance until late layers, onset shifting earlier as scale grows. Null: all tiers "
             "flat at chance inside the shuffle band.")


def fig5_accuracy_grid(commitments: dict, cfg: Config) -> None:
    """Supplementary — pilot accuracy per cell across the full 30-cell grid,
    with out-of-band covariates (outside 25–80% target band) marked with dotted borders."""
    apply_style(cfg)
    if not commitments:
        return
    models, tiers = cfg.active_models(), cfg.active_tiers()
    M = np.full((len(models), len(tiers)), np.nan)
    for v in commitments.values():
        if v["model"] in models and v["tier"] in tiers:
            M[models.index(v["model"]), tiers.index(v["tier"])] = v["accuracy"]
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH * 0.78, cfg.FIG_WIDTH * 0.42))
    im = ax.imshow(M, cmap=CMAP_SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(len(models)):
        for j in range(len(tiers)):
            if not np.isfinite(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color=INK3, fontsize=8)
                continue
            cid = cell_id(models[i], tiers[j])
            v = commitments.get(cid, {})
            in_band = v.get("in_band", False) if "in_band" in v else (cfg.ACCURACY_BAND[0] <= M[i, j] <= cfg.ACCURACY_BAND[1])
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.4,
                    color=SURFACE if M[i, j] > 0.55 else INK,
                    fontweight="bold" if in_band else "normal")
            if not in_band:
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False,
                                           edgecolor=PALETTE[7], lw=1.5, ls=":"))
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"{t}\n{TIER_SPECS[t]['family'][:4]}" for t in tiers])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.replace("qwen2.5-", "") for m in models])
    ax.set_title(f"Pilot accuracy per cell (bold = in {cfg.ACCURACY_BAND[0]:.0%}–{cfg.ACCURACY_BAND[1]:.0%} target band; "
                 f"dotted = out-of-band covariate)")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, label="pilot accuracy", fraction=0.035, pad=0.02)
    fig.tight_layout()
    save_fig(fig, "fig5_cell_commitment_grid", cfg,
             "Supplementary — pilot accuracy per (model × tier) cell. All 30 cells are committed "
             "to enable full-grid hierarchical modeling; dotted outlines indicate cells outside "
             f"the {cfg.ACCURACY_BAND[0]:.0%}–{cfg.ACCURACY_BAND[1]:.0%} band which are controlled statistically as covariates.")


def fig6_correlations(corr: pd.DataFrame, cfg: Config) -> None:
    """Supplementary — pairwise signal correlation, raw Spearman per cell."""
    apply_style(cfg)
    d = corr[(corr.model != "POOLED") & corr.spearman_raw.notna()]
    if not len(d):
        return
    piv = d.pivot_table(index=["model", "tier"], columns="pair", values="spearman_raw")
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH * 0.72, max(2.4, 0.22 * len(piv))))
    im = ax.imshow(piv.values, cmap=CMAP_DIV, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=20, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{m.replace('qwen2.5-', '')}·{t}" for m, t in piv.index], fontsize=6.8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.4,
                        color=SURFACE if abs(v) > 0.55 else INK)
    ax.set_title("Pairwise Spearman between raw signals (test split)")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.03, pad=0.02)
    fig.tight_layout()
    save_fig(fig, "fig6_signal_correlations", cfg,
             "Supplementary — raw-score rank correlation between the three signals per cell.")


def stage_figures(signals: pd.DataFrame, sweep: pd.DataFrame, h1: dict, h3: dict,
                  abst: dict, corr: pd.DataFrame, commitments: dict, cfg: Config) -> None:
    for fn, args in (
        (fig1_calibration, (signals, h1, cfg)),
        (fig2_quadrant, (cfg, abst)),
        (fig3_model_delta, (h3, cfg)),
        (fig4_depth_curves, (sweep, cfg)),
        (fig5_accuracy_grid, (commitments, cfg)),
        (fig6_correlations, (corr, cfg)),
    ):
        try:
            fn(*args)
        except Exception as exc:                          # noqa: BLE001
            LOG.log("figure_failed", figure=fn.__name__, error=str(exc)[:200])
# %%
# ============================================================================
# CELL 20 — table export (CSV + LaTeX), research-paper shaped
# ============================================================================


def export_table(df: pd.DataFrame, name: str, cfg: Config, caption: str = "",
                 float_fmt: str = "%.3f") -> None:
    if df is None or not len(df):
        return
    df.to_csv(PATHS["tables"] / f"{name}.csv", index=False)
    try:
        df.to_parquet(PATHS["tables"] / f"{name}.parquet", index=False)
    except Exception:                                     # noqa: BLE001
        pass
    if cfg.LATEX_TABLES:
        try:
            tex = df.to_latex(index=False, float_format=float_fmt, escape=True,
                              caption=caption or name.replace("_", " "), label=f"tab:{name}")
            (PATHS["tables"] / f"{name}.tex").write_text(tex)
        except Exception:                                 # noqa: BLE001
            pass
    LOG.log("table_saved", name=name, rows=len(df), cols=len(df.columns))


def stage_tables(bank: dict, graded: pd.DataFrame, signals: pd.DataFrame, sweep: pd.DataFrame,
                 entropy: pd.DataFrame, corr: pd.DataFrame, commitments: dict, h0: dict,
                 h1: dict, h2: dict, h3: dict, h4: dict, abst: dict, omni: pd.DataFrame,
                 cfg: Config) -> None:
    # T1 — dataset composition
    export_table(pd.DataFrame([
        dict(tier=t, label=TIER_SPECS[t]["label"], family=TIER_SPECS[t]["family"],
             answer_form=TIER_SPECS[t]["answer_form"], difficulty=TIER_SPECS[t]["difficulty"],
             source=TIER_SPECS[t]["hf_id"], n=len(bank.get(t, [])),
             n_train=sum(r["split"] == "train" for r in bank.get(t, [])),
             n_cal=sum(r["split"] == "calibration" for r in bank.get(t, [])),
             n_test=sum(r["split"] == "test" for r in bank.get(t, [])),
             max_new_tokens=TIER_SPECS[t]["max_new_tokens"])
        for t in cfg.active_tiers()]), "t1_dataset_composition", cfg,
        "Six-tier retrieval→reasoning ladder with per-cell splits.")

    # T2 — cell commitment (the ragged grid)
    export_table(pd.DataFrame(list(commitments.values())), "t2_cell_commitment", cfg,
                 "Pilot accuracy and 25–80% band commitment verdict per cell.")

    # T3 — parse / grading reliability
    if len(graded):
        g = graded.groupby(["model", "tier", "variant"]).agg(
            n=("qid", "size"), parse_rate=("parse_ok", "mean"),
            accuracy=("correct", "mean"), resolved_rate=("resolved", "mean")).reset_index()
        export_table(g, "t3_parse_and_accuracy", cfg,
                     "Format-compliance and accuracy per (model, tier, elicitation format).")
        export_table(graded.groupby(["answer_form", "grader"]).size().reset_index(name="n"),
                     "t4_grader_tier_usage", cfg,
                     "Which grading tier resolved each item — the LLM-judge-free audit trail.")

    # T5 — H0 / Gate 2
    export_table(pd.DataFrame([{"pair": k, **v} for k, v in h0.get("pairs", {}).items()]),
                 "t5_h0_format_agreement", cfg,
                 "Pairwise Spearman across verbalized formats A/B/C (Gate 2).")

    # T6 — H1 / Murphy
    export_table(pd.DataFrame([{"signal": s, **v} for s, v in h1.get("pooled", {}).items()]),
                 "t6_h1_murphy_decomposition", cfg,
                 "Per-signal ECE, Brier and Murphy components on the test split.")

    # T7 — probe sweep + T8 depth onsets
    export_table(sweep, "t7_probe_sweep", cfg,
                 "Probe AUROC per (cell × layer percentile) with shuffle-null and surface baselines.")
    if h4.get("available"):
        export_table(pd.DataFrame(h4["onsets"]), "t8_h4_depth_onsets", cfg,
                     "Onset percentile (first layer reaching the AUROC gate) per tier and model.")

    # T9 — correlations, T10 — Omniscience-Index, T11 — abstention
    export_table(corr, "t9_signal_correlations", cfg,
                 "Spearman on raw scores and Pearson on calibrated scores.")
    export_table(omni, "t10_omniscience_index", cfg,
                 "Decision-level Omniscience-Index per model and tier (PLAN §8.2).")
    export_table(pd.DataFrame([{"cell": k, **v} for k, v in abst.get("per_cell", {}).items()]),
                 "t11_abstention_split", cfg,
                 "Justified hedge vs missed knowledge for every Format C Pass (PLAN §4.1).")

    # T12 — per-question master table (the paper's data appendix)
    if len(signals):
        export_table(signals, "t12_per_question_signals", cfg,
                     "Per-question raw and calibrated values for all three signals.")
    if len(entropy):
        export_table(entropy, "t13_semantic_entropy", cfg,
                     "Semantic-entropy cluster statistics per question.")

    # T14 — hypothesis verdict summary
    export_table(pd.DataFrame([
        dict(hypothesis="H0", gate="Gate 2", passed=h0.get("gate2_pass"), verdict=h0.get("verdict")),
        dict(hypothesis="H1", gate="—", passed=h1.get("h1_pass"), verdict=h1.get("verdict")),
        dict(hypothesis="H2", gate="Gate 1", passed=h2.get("h2_pass"), verdict=h2.get("verdict")),
        dict(hypothesis="H3", gate="Gate 4", passed=h3.get("h3_pass"), verdict=h3.get("verdict")),
        dict(hypothesis="H4", gate="Gate 3", passed=h4.get("h4_pass"), verdict=h4.get("verdict")),
    ]), "t15_hypothesis_verdicts", cfg, "Pre-registered hypothesis verdicts.")


# %%
# ============================================================================
# CELL 21 — compute ledger (X2) + §17.2 run-log report
# ============================================================================


def compute_ledger(cfg: Config) -> pd.DataFrame:
    """Actual GPU seconds spent per (model, tier, stage), from the event log —
    the X2 ledger PLAN §10 asks for, measured rather than estimated."""
    rows = []
    for ev in jsonl_read(PATHS["logs"] / "events.jsonl"):
        if ev.get("event") == "stage_timing":
            rows.append({k: ev.get(k) for k in ("model", "tier", "stage", "seconds", "generated",
                                                "out_tokens", "parse_rate")})
    df = pd.DataFrame(rows)
    if len(df):
        df["gpu_hours"] = df["seconds"] / 3600.0
        df.to_parquet(PATHS["derived"] / "compute_ledger.parquet", index=False)
    return df


def stage_report(bank: dict, commitments: dict, h0: dict, h1: dict, h2: dict, h3: dict,
                 h4: dict, gate3: dict, judge: dict, sanity: dict, hlr: dict, cfg: Config,
                 popularity: "dict | None" = None, matched: "dict | None" = None) -> dict:
    ledger = compute_ledger(cfg)
    total_h = float(ledger["gpu_hours"].sum()) if len(ledger) else 0.0
    committed = [k for k, v in commitments.items() if v["committed"]]
    gates = {
        "gate1_grading_sanity": {
            # Human sheet is authoritative; the judge is a second opinion.
            # `pass: null` means UNVERIFIED, not passed.
            "pass": judge.get("gate1_pass"),
            "source": ("human_manual_sheet" if (judge.get("manual") or {}).get("available")
                       else "automated_judge" if judge.get("agreement_overall") is not None
                       else "none"),
            "automated_agreement": judge.get("agreement_overall"),
            "manual_agreement": (judge.get("manual") or {}).get("agreement"),
            "manual_rows_labelled": (judge.get("manual") or {}).get("n_labelled", 0),
            "manual_rows_total": (judge.get("manual") or {}).get("n_rows", 0),
            "manual_disagreements": (judge.get("manual") or {}).get("n_disagreements", 0),
            "manual_by_grader": (judge.get("manual") or {}).get("by_grader", {}),
            "threshold": cfg.GATE1_AGREEMENT,
            "note": ("Gate 1 closes on the HUMAN check sheet (PLAN §16). A null verdict means "
                     "the sheet is unfilled — it must not be reported as passed."),
        },
        "gate2_format_agreement": {"pass": h0.get("gate2_pass"),
                                   "min_lower_ci": min([v["lo"] for v in h0.get("pairs", {}).values()
                                                        if np.isfinite(v.get("lo", np.nan))] or [np.nan])},
        "gate3_probe_validity": {
            "cells": len(gate3), "passed": sum(v["passes"] for v in gate3.values()),
            "dirty_activation_cells": sum(1 for v in gate3.values() if not v["activations_clean"]),
            "p0_negative_control_failures": sum(1 for v in gate3.values()
                                                if not v.get("p0_neg_control_pass")),
            "enforced": cfg.GATE3_ENFORCE,
        },
        "gate4_model_comparison": {"pass": h3.get("h3_pass"), "available": h3.get("available")},
    }
    report = {
        "provenance": PROV,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "grid": {"models": cfg.active_models(), "tiers": cfg.active_tiers(),
                 "cells_total": len(commitments), "cells_committed": len(committed),
                 "ragged_by": len(commitments) - len(committed)},
        "questions": {t: len(v) for t, v in bank.items()},
        "gates": gates,
        "hypotheses": {"H0": h0.get("verdict"), "H1": h1.get("verdict"), "H2": h2.get("verdict"),
                       "H3": h3.get("verdict"), "H4": h4.get("verdict")},
        # Qualifiers that change how a verdict should be read. Gate 3 removes a
        # failing cell's internal signal, which shrinks the grid H1 compares
        # over and the grid H4's depth curves are drawn from; a bare verdict
        # string does not show that.
        "hypothesis_qualifiers": {
            "H1_like_for_like": (h1.get("coverage") or {}).get("like_for_like"),
            "H1_coverage_warning": (h1.get("coverage") or {}).get("warning"),
            "H1_cells_all_three_signals": (h1.get("coverage") or {}).get("cells_all_three_signals", []),
            "H2_failing_conjuncts": [k for k, v in (h2.get("conjuncts") or {}).items() if not v],
            "H2_gate3_cells_failing": h2.get("gate3_cells_failing", []),
            "H3_excluded_missing_signal": h3.get("n_excluded_missing_signal"),
            "H4_gate3_excluded_cells": h4.get("gate3_excluded_cells", []),
            "H4_coverage_note": h4.get("coverage_note"),
            "internal_cells_skipped": sorted(
                (json_read(PATHS["derived"] / "internal_gate3_skipped.json", {}) or {})),
        },
        "canonical_format": (json_read(PATHS["derived"] / "calibration_meta.json", {}) or {})
                            .get("canonical_format"),
        "entropy_sanity": sanity,
        "hierarchical_regression": {k: hlr.get(k) for k in
                                    ("method", "n", "pseudo_r2", "converged",
                                     "random_intercepts_requested", "random_intercepts_fitted",
                                     # without these, a failed fit reads as a
                                     # successful one with a null pseudo-R2
                                     "error", "reason", "available",
                                     "signals_dropped_all_missing",
                                     "signals_near_all_missing")},
        # The base-rate control that replaced cell deletion, and the popularity
        # manipulation the deletion had erased (AUDIT findings 5 and 9). These
        # wrote their own JSON but appeared in no report, so a reader of
        # final_report.json could not tell they had run.
        "base_rate_control": {
            "band_mode": ("covariate" if cfg.COMMIT_CELLS_OUTSIDE_BAND else "deletion"),
            "accuracy_band": list(cfg.ACCURACY_BAND),
            "cells_out_of_band": sum(1 for v in commitments.values() if not v.get("in_band")),
            "difficulty_matched": {"available": (matched or {}).get("available", False),
                                   "tiers": sorted((matched or {}).get("per_tier", {})),
                                   "artifact": "derived/difficulty_matched.json"},
        },
        "popularity_gradient": {
            "available": (popularity or {}).get("available", False),
            "tiers_covered": (popularity or {}).get("tiers_covered", []),
            "coverage_warning": (popularity or {}).get("coverage_warning"),
            "contrasts": (popularity or {}).get("contrasts", {}),
            "artifact": "derived/popularity_gradient.json",
        },
        "compute": {"measured_gpu_hours": round(total_h, 3),
                    "by_model": (ledger.groupby("model")["gpu_hours"].sum().round(3).to_dict()
                                 if len(ledger) else {})},
        "judge": {k: judge.get(k) for k in ("enabled", "judge_model", "n_decided",
                                            "agreement_overall", "agreement_by_grader")},
        "artifacts": {"figures": sorted(p.name for p in PATHS["figures"].glob("*.png")),
                      "tables": sorted(p.name for p in PATHS["tables"].glob("*.csv")),
                      "derived": sorted(p.name for p in PATHS["derived"].glob("*"))},
    }
    json_write(PATHS["meta"] / "final_report.json", report)

    # §17.2 run-log rows, ready to paste into PLAN.md.
    # PLAN §0 still says "no runs yet" and §17.2 is empty after three runs
    # (AUDIT finding 8). Provenance is emitted here in paste-ready form so
    # closing that gap is a copy, not a reconstruction from memory.
    unbound = str(PROV.get("code_sha", "")).startswith("UNBOUND")
    lines = [
        f"### Run `{cfg.RUN_NAME}` — {report['finished_utc']}",
        "",
        f"- code_sha    : `{PROV.get('code_sha')}`"
        + ("  **UNBOUND — this run is not traceable to a commit**" if unbound else ""),
        f"- config_hash : `{PROV.get('config_hash')}`",
        f"- seed        : {PROV.get('seed')}",
        f"- platform    : {PROV.get('platform')}  ·  dtype {PROV.get('dtype')}",
        f"- torch {PROV.get('torch')} · transformers {PROV.get('transformers')} "
        f"· math_verify {PROV.get('math_verify')}",
        f"- grid        : {len(committed)}/{len(commitments)} cells committed",
        f"- GPU-hours   : {round(total_h, 3)}",
        f"- canonical verbal format: {report.get('canonical_format')}",
        "",
        "| Run-log ID | What it did | Outcome | Headline |",
        "|---|---|---|---|",
    ]
    for hid, verdict in report["hypotheses"].items():
        if verdict:
            lines.append(f"| {cfg.RUN_NAME}_{hid} | {hid} per PLAN §13 | "
                         f"{'pass' if 'supported' in str(verdict) else 'null/falsified'} | {verdict} |")
    (PATHS["meta"] / "run_log_rows.md").write_text("\n".join(lines))
    if unbound:
        LOG.log("provenance_unbound", note="code_sha is UNBOUND — set CODE_SHA before a production run")
    LOG.log("report_written", cells_committed=len(committed), gpu_hours=round(total_h, 2))
    return report


# %%
# ============================================================================
# CELL 22 — MAIN driver
# Per model: run every generation stage, checkpoint, then FREE GPU MEMORY.
# ============================================================================


def timed(stage: str, model: str, fn: Callable, *args) -> dict:
    t0 = time.time()
    res = fn(*args)
    for tier_key, st in (res or {}).items():
        if isinstance(st, dict):
            LOG.log("stage_timing", stage=stage, model=model, tier=str(tier_key).split(":")[0],
                    seconds=st.get("seconds", 0.0), generated=st.get("generated", 0),
                    out_tokens=st.get("out_tokens", 0), parse_rate=round(st.get("parse_rate", 0.0), 4))
    LOG.log("stage_done", stage=stage, model=model, secs=round(time.time() - t0, 1))
    return res


def run_generation_stages_for_model(model: str, bank: dict, committed: set[str],
                                    cfg: Config, tiers: Sequence[str] | None = None,
                                    purge: bool = False) -> None:
    """Load one model, run every enabled generation stage, then tear it down.

    `tiers` restricts this worker to a slice of the ladder — that is how
    MODEL_REPLICAS shards work across replicas of the same weights.

    `purge` deletes the on-disk snapshot afterwards. It must stay False on the
    pilot pass: the band gate needs every model's pilot before any cell can be
    committed, so each model is loaded twice, and purging after pass 1 would
    force a full re-download for pass 2.
    """
    sub = replace(cfg, ONLY_TIERS=tuple(tiers)) if tiers else cfg
    lm = None
    try:
        lm = load_model(model, cfg)
        if "pilot" in cfg.STAGES:
            timed("pilot", model, stage_pilot, lm, bank, sub)
        if "verbal" in cfg.STAGES:
            timed("verbal", model, stage_verbal, lm, bank, sub, committed)
        if "forced" in cfg.STAGES:
            timed("forced", model, stage_forced, lm, bank, sub, committed)
        if "sample" in cfg.STAGES:
            timed("sample", model, stage_sample, lm, bank, sub, committed)
        if "extract" in cfg.STAGES:
            timed("extract", model, stage_extract, lm, bank, sub, committed)
    except Exception as exc:                              # noqa: BLE001
        LOG.log("model_failed", model=model, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # GPU memory is always cleared here; disk is only reclaimed when the
        # caller says this was the model's final pass.
        free_model(lm, cfg, purge=purge)


def replica_tier_shards(tiers: list[str], n: int) -> list[list[str]]:
    return [tiers[i::n] for i in range(min(n, len(tiers)))]


def run_pipeline(cfg: Config, judge_cfg: JudgeConfig) -> dict:
    t_start = time.time()
    LOG.log("pipeline_start", stages=list(cfg.STAGES), models=cfg.active_models())

    # ---- shared question bank (model-independent, built once) -------------
    bank = build_question_bank(cfg) if "data" in cfg.STAGES else json_read(
        PATHS["data"] / "question_bank.json", {})
    if not bank:
        raise RuntimeError("no question bank — run the 'data' stage first")

    nli = NLIGrader(cfg.NLI_MODEL, cfg.NLI_ENTAIL_THRESHOLD, cfg.NLI_BATCH_SIZE,
                    cfg.resolved_dtype(), "cuda" if DEVICES["cuda"] else "cpu") \
        if cfg.USE_NLI_FALLBACK else None

    # ---- pass 1: pilots, then the band gate decides the ragged grid -------
    commitments = json_read(PATHS["derived"] / "cell_commitments.json", {})
    if "pilot" in cfg.STAGES:
        for model in cfg.active_models():
            run_generation_stages_for_model(
                model, bank, set(), replace(cfg, STAGES=("pilot",)))
        commitments = evaluate_band_gate(bank, cfg, nli)
    committed = {k for k, v in commitments.items() if v.get("committed")}
    if not committed and any(s in cfg.STAGES for s in ("sample", "extract")):
        LOG.log("no_committed_cells", note="band gate excluded every cell; "
                "set COMMIT_CELLS_OUTSIDE_BAND=True to override")

    # ---- pass 2: the heavy generation stages, per model wave --------------
    gen_stages = tuple(s for s in cfg.STAGES if s in ("verbal", "forced", "sample", "extract"))
    if gen_stages:
        gcfg = replace(cfg, STAGES=gen_stages)
        for wave in plan_model_batches(cfg.active_models(), cfg):
            shards = replica_tier_shards(cfg.active_tiers(), cfg.MODEL_REPLICAS) \
                if cfg.MODEL_REPLICAS > 1 else [None]
            jobs = [(m, sh) for m in wave for sh in shards]
            # This is each model's FINAL pass, so disk may be reclaimed here.
            # Never with replicas/shards: the last shard to finish would delete
            # a snapshot its siblings are still reading.
            purge = (cfg.PURGE_WEIGHTS == "after_model" and len(jobs) == len(wave))
            if len(jobs) == 1:
                run_generation_stages_for_model(jobs[0][0], bank, committed, gcfg,
                                                jobs[0][1], purge=purge)
            else:
                LOG.log("wave_start", models=wave, replicas=cfg.MODEL_REPLICAS, jobs=len(jobs))
                with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
                    futs = [ex.submit(run_generation_stages_for_model, m, bank, committed,
                                      gcfg, sh, purge)
                            for m, sh in jobs]
                    for f in futs:
                        f.result()
            free_cuda()

    # ---- analysis (CPU + the small NLI model only) ------------------------
    graded = stage_grade(bank, cfg, nli) if "grade" in cfg.STAGES else \
        pd.DataFrame(jsonl_read(PATHS["derived"] / "graded.jsonl"))
    sanity = entropy_sanity_check(cfg)
    entropy = stage_entropy(bank, cfg, nli) if "entropy" in cfg.STAGES else \
        (pd.read_parquet(PATHS["derived"] / "entropy.parquet")
         if (PATHS["derived"] / "entropy.parquet").exists() else pd.DataFrame())
    if nli is not None:
        nli.free()

    sweep = stage_probe(bank, graded, entropy, commitments, cfg) if "probe" in cfg.STAGES else \
        (pd.read_parquet(PATHS["derived"] / "probe_sweep.parquet")
         if (PATHS["derived"] / "probe_sweep.parquet").exists() else pd.DataFrame())
    # Only trust the blocked-cell list when the probe stage actually produced
    # it this run; on a stats-only rerun the file on disk describes a different
    # set of activations.
    gate3 = gate3_verdict(sweep, cfg,
                          blocked=(json_read(PATHS["derived"] / "gate3_blocked_cells.json", {})
                                   if "probe" in cfg.STAGES else {}))

    signals, meta = assemble_signals(bank, graded, entropy, sweep, commitments, cfg, gate3=gate3) \
        if "calibrate" in cfg.STAGES else (pd.DataFrame(), {"verbal_long": pd.DataFrame()})

    # ---- Gate 1 BEFORE the hypothesis tests ------------------------------
    # PLAN §16 conditions H2 on "AND Gate 1 holds", so the grading-sanity
    # verdict has to exist before H2 is evaluated. The judge used to run after
    # the stats block, which is why H2's pass rule silently dropped the
    # conjunct: the value simply was not available yet (AUDIT finding 8).
    judge = stage_judge(bank, graded, judge_cfg, cfg) if judge_cfg.ENABLED else {"enabled": False}
    if len(graded):
        sheet_path = export_manual_check_sheet(bank, graded, cfg)
        manual = ingest_manual_sheet(sheet_path, cfg)
        judge["manual"] = manual
        judge["gate1_pass"] = combined_gate1(judge, manual, cfg)

    h0 = h1 = h2 = h3 = h4 = hlr = {}
    popularity = matched = {}
    abst = {"by_category": {}, "per_cell": {}}
    corr, omni = pd.DataFrame(), pd.DataFrame()
    if "stats" in cfg.STAGES and len(signals):
        h0 = test_h0_format_agreement(meta["verbal_long"], cfg)
        h1 = test_h1_signal_calibration(signals, cfg)
        h2 = test_h2_quadrants(signals, bank, cfg,
                               gate1_pass=judge.get("gate1_pass"), gate3=gate3)
        abst = abstention_split(graded, cfg)
        omni = omniscience_index(graded)
        h3 = test_h3_base_vs_instruct(signals, abst, cfg)
        h4 = test_h4_depth(sweep, cfg, gate3=gate3)
        hlr = hierarchical_regression(signals, cfg, bank=bank)
        popularity = popularity_contrast(signals, graded, cfg)
        matched = difficulty_matched_view(signals, cfg)
        corr = correlation_table(signals, cfg)

    if "figures" in cfg.STAGES:
        # Figure 4 is the depth-curve figure; it must show the same cells H4
        # was computed over, not the unfiltered sweep.
        sweep_fig, _ = gate3_filter_sweep(sweep, gate3, cfg)
        stage_figures(signals, sweep_fig, h1, h3, abst, corr, commitments, cfg)
    if "tables" in cfg.STAGES:
        stage_tables(bank, graded, signals, sweep, entropy, corr, commitments,
                     h0, h1, h2, h3, h4, abst, omni, cfg)
    report = stage_report(bank, commitments, h0, h1, h2, h3, h4, gate3, judge, sanity, hlr, cfg,
                          popularity=popularity, matched=matched) \
        if "report" in cfg.STAGES else {}

    free_cuda()
    if cfg.PURGE_WEIGHTS == "after_run":
        purge_all_weights()
    LOG.log("pipeline_done", minutes=round((time.time() - t_start) / 60, 1))
    return {"bank": bank, "commitments": commitments, "graded": graded, "entropy": entropy,
            "sweep": sweep, "signals": signals, "h0": h0, "h1": h1, "h2": h2, "h3": h3,
            "h4": h4, "hlr": hlr, "corr": corr, "omni": omni, "abstention": abst,
            "popularity": popularity, "difficulty_matched": matched,
            "gate3": gate3, "judge": judge, "report": report}


# %%
# ============================================================================
# CELL 23 — RUN
# Safe to re-execute: every stage is idempotent and resumes from checkpoints.
# ============================================================================
RESULTS = run_pipeline(CFG, JUDGE)

print("\n" + "=" * 74)
print(f"run '{CFG.RUN_NAME}'  ·  config {CFG.hash()}  ·  output {PATHS['root']}")
print("=" * 74)
_rep = RESULTS.get("report", {})
if _rep:
    _g = _rep["grid"]
    print(f"grid          : {_g['cells_committed']}/{_g['cells_total']} cells committed "
          f"({_g['ragged_by']} excluded by the 25–80% band)")
    print(f"gpu measured  : {_rep['compute']['measured_gpu_hours']} hours")
    print("\ngates")
    for k, v in _rep["gates"].items():
        print(f"  {k:26s} {v}")
    print("\nhypotheses")
    for k, v in _rep["hypotheses"].items():
        print(f"  {k}: {v}")
    print(f"\nfigures : {len(_rep['artifacts']['figures'])}  ->  {PATHS['figures']}")
    print(f"tables  : {len(_rep['artifacts']['tables'])}  ->  {PATHS['tables']}")
    print(f"\nNEXT: fill in {PATHS['tables'] / 'gate1_manual_check_sheet.csv'} to close Gate 1,")
    print(f"      then paste {PATHS['meta'] / 'run_log_rows.md'} into PLAN.md §17.2.")
