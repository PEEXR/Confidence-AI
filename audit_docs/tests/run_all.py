#!/usr/bin/env python3
"""Run every audit-repair test. Exit non-zero if any fails.

    python audit_docs/tests/run_all.py

Needs numpy, pandas, scipy, scikit-learn, statsmodels, torch (CPU is fine) and
math_verify. Each test targets a specific finding in ../AUDIT.md.
"""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = [
    ("test_tap.py", "AUDIT finding 1 — activation tap reads the last PROMPT token"),
    ("test_logprobs.py", "AUDIT finding 3 — sequence log-probs, against a real generate()"),
    ("test_entropy.py", "AUDIT finding 3 — math clustering, weighting, entropy normalisation"),
    ("test_analysis.py", "AUDIT findings 2, 4, 6, 7, 8, 9 — verbal axis, H2/H3, gates, grid"),
]
fails = []

# The .ipynb is what runs on molab/Kaggle. An unsynced repo runs the OLD code
# while the diff shows the new — silent and expensive, so it is gated here.
print("\n=== notebook sync — confidence_pipeline.ipynb matches the .py", flush=True)
if subprocess.run([sys.executable, str(HERE.parent / "sync_notebook.py"), "--check"],
                  cwd=HERE.parents[1]).returncode != 0:
    fails.append("notebook sync")

for name, what in TESTS:
    print(f"\n=== {name} — {what}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / name)], cwd=HERE.parents[1])
    if r.returncode != 0:
        fails.append(name)
print("\n" + "=" * 70)
n_checks = len(TESTS) + 1
print(f"{n_checks - len(fails)}/{n_checks} checks passed")
if fails:
    print("FAILED:", ", ".join(fails))
sys.exit(1 if fails else 0)
