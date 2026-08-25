# -*- coding: utf-8 -*-
"""
Run unified analysis pipeline on all 5 models (0.8B, 2B, 4B, 9B-Instruct, 9B-Base).
Grades generations, runs semantic entropy, depth probe regressions, calibrations,
H0-H4 hypothesis testing, and generates all Figures (1-5) and Tables (1-15).
"""
import os
import sys
import types
import importlib.machinery

os.environ["INSTALL_LINEAR_ATTN_KERNELS"] = "0"

# Mock datasets
ds = types.ModuleType("datasets")
ds.__spec__ = importlib.machinery.ModuleSpec("datasets", None)
ds.load_dataset = lambda *args, **kwargs: None
sys.modules["datasets"] = ds

# Mock transformers
tr = types.ModuleType("transformers")
tr.__spec__ = importlib.machinery.ModuleSpec("transformers", None)
for name in ("AutoTokenizer", "AutoModelForCausalLM", "AutoConfig", "AutoModelForSequenceClassification", "AutoModelForImageTextToText"):
    setattr(tr, name, type(name, (), {"from_pretrained": lambda *args, **kwargs: None}))
sys.modules["transformers"] = tr

# Mock accelerate
acc = types.ModuleType("accelerate")
acc.__spec__ = importlib.machinery.ModuleSpec("accelerate", None)
sys.modules["accelerate"] = acc

# Mock google
g = types.ModuleType("google")
g.__spec__ = importlib.machinery.ModuleSpec("google", None, is_package=True)
g.__path__ = []
sys.modules["google"] = g

# Mock torch
tm = types.ModuleType("torch")
tm.__spec__ = importlib.machinery.ModuleSpec("torch", None)
class Tensor: pass
tm.Tensor = Tensor
tm.cuda = types.ModuleType("cuda")
tm.cuda.is_available = lambda: False
tm.cuda.device_count = lambda: 0
tm.cuda.is_bf16_supported = lambda: False
tm.cuda.empty_cache = lambda: None
tm.cuda.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})
tm.__version__ = "cpu-analysis"
tm.float32 = "float32"
tm.bfloat16 = "bfloat16"
tm.float16 = "float16"
tm.manual_seed = lambda s: None
tm.no_grad = lambda: (lambda f: f)
sys.modules["torch"] = tm

import gc
import json
import re
import shutil
import time
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("=" * 74)
print("STARTING 5-MODEL UNIFIED ANALYSIS & FIGURE GENERATION")
print("=" * 74)

# Import functions from confidence_pipeline
import confidence_pipeline as cp

# Disable math_verify subprocesses on Windows — use in-memory SymPy and normalisation
cp.HAS_MATH_VERIFY = False

# Configure for all 5 models
all_models = (
    "qwen3.5-0.8b-instruct",
    "qwen3.5-2b-instruct",
    "qwen3.5-4b-instruct",
    "qwen3.5-9b-instruct",
    "qwen3.5-9b-base"
)

cfg = replace(
    cp._BASE_CFG,
    RUN_NAME="all_5models",
    OUTPUT_ROOT="./confidence_out",
    ONLY_MODELS=all_models,
    USE_NLI_FALLBACK=False,  # deterministic string/latex/numeric grading on CPU
    STAGES=(
        "grade",
        "entropy",
        "probe",
        "calibrate",
        "stats",
        "figures",
        "tables",
        "report"
    )
)

# Update paths in cp module
cp.CFG = cfg
cp.PATHS = cp._resolve_paths(cfg)

# Load shared question bank
bank = cp.json_read(cp.PATHS["data"] / "question_bank.json")
if not bank:
    bank = cp.json_read(Path("./confidence_out/s1/data/question_bank.json"))
    cp.json_write(cp.PATHS["data"] / "question_bank.json", bank)
    shutil.copy2(Path("./confidence_out/s1/data/bank_manifest.json"), cp.PATHS["data"] / "bank_manifest.json")

print(f"Loaded question bank: {list(bank.keys())}")
print(f"Active models ({len(cfg.active_models())}): {cfg.active_models()}")
print(f"Active tiers ({len(cfg.active_tiers())}): {cfg.active_tiers()}")

# 1. Stage Grade
print("\n--- 1. Running Stage Grade ---")
graded = cp.stage_grade(bank, cfg, nli=None)
print(f"Grading complete: {len(graded):,} rows graded.")
gc.collect()

# 2. Stage Entropy
print("\n--- 2. Running Stage Entropy ---")
sanity = cp.entropy_sanity_check(cfg)
entropy = cp.stage_entropy(bank, cfg, nli=None)
print(f"Semantic entropy complete: {len(entropy):,} rows.")
gc.collect()

# 3. Stage Probe
print("\n--- 3. Running Stage Probe (5-depth sweeps across all 5 models) ---")
commitments = cp.json_read(cp.PATHS["derived"] / "cell_commitments.json", {})
sweep = cp.stage_probe(bank, graded, entropy, commitments, cfg)
gate3 = cp.gate3_verdict(sweep, cfg)
print(f"Probes complete: {len(sweep)} depth points evaluated.")
gc.collect()

# 4. Stage Calibrate & Assemble Signals
print("\n--- 4. Assembling Signals & Calibrating ---")
signals, meta = cp.assemble_signals(bank, graded, entropy, sweep, commitments, cfg)
print(f"Signals assembled: {len(signals):,} test rows.")
gc.collect()

# 5. Stage Stats & Hypothesis Tests (H0 - H4)
print("\n--- 5. Testing Hypotheses (H0-H4) & Computing Statistics ---")
feats = cp.question_features(bank)
h0 = cp.test_h0_format_agreement(meta["verbal_long"], cfg)
h1 = cp.test_h1_signal_calibration(signals, cfg)
h2 = cp.test_h2_quadrants(signals, bank, cfg, gate1_pass=True, gate3=gate3)
abst = cp.abstention_split(graded, cfg)
omni = cp.omniscience_index(graded)

elic_ok = None
base_names = [m for m, s in cp.MODEL_SPECS.items() if s.get("rung") == "h3-comparison"]
if len(graded) and base_names:
    bg = graded[(graded.model.isin(base_names)) & (graded.variant == "FORCED")]
    if len(bg):
        elic_ok = bool(bg["parse_ok"].astype(float).mean() >= cfg.BASE_ELICITATION_MIN_PARSE_RATE)

h3 = cp.test_h3_base_vs_instruct(signals, abst, cfg, gate4_elicit_ok=elic_ok)
h4 = cp.test_h4_depth(sweep, cfg)
hlr = cp.hierarchical_regression(signals, cfg, feats=feats)
corr = cp.correlation_table(signals, cfg)
pop = cp.popularity_contrast(signals, cfg)
dmatch = cp.difficulty_matched_view(signals, graded, cfg)

if len(graded):
    cp.export_manual_check_sheet(bank, graded, cfg)
manual_sheet = cp.ingest_manual_sheet(cfg)

# 6. Stage Figures
print("\n--- 6. Generating Figures 1-5 ---")
cp.stage_figures(signals, sweep, h1, h3, abst, corr, commitments, cfg)
figs_created = sorted(list(cp.PATHS["figures"].glob("*.png")))
print(f"Created {len(figs_created)} figures in {cp.PATHS['figures']}")
for f in figs_created:
    print(f"  [Figure] {f.name}")

# 7. Stage Tables
print("\n--- 7. Exporting Tables t1-t15 ---")
cp.stage_tables(bank, graded, signals, sweep, entropy, corr, commitments,
                h0, h1, h2, h3, h4, abst, omni, cfg)
tables_created = sorted(list(cp.PATHS["tables"].glob("*.csv")))
print(f"Created {len(tables_created)} tables in {cp.PATHS['tables']}")
for t in tables_created:
    print(f"  [Table] {t.name}")

# 8. Stage Report
print("\n--- 8. Writing Final Report ---")
judge = {"enabled": False}
report = cp.stage_report(bank, commitments, h0, h1, h2, h3, h4, gate3, judge, sanity,
                         hlr, manual_sheet, cfg)

print("\n" + "=" * 74)
print(f"Unified 5-Model Run Complete! Output directory: {cp.PATHS['root']}")
print("=" * 74)
print("\nHypothesis Verdicts:")
for k, v in report.get("hypotheses", {}).items():
    print(f"  {k}: {v}")
