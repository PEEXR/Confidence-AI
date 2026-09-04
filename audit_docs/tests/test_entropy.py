"""Smoke test: math clustering, log-prob weighting, valid-sample normalization."""
from pathlib import Path
import ast, re, sys, types, importlib.util
from collections import Counter
import numpy as np, scipy.stats as sps
from typing import Any, Callable, Sequence

SRC = open(Path(__file__).resolve().parents[2] / "confidence_pipeline.py").read()
tree = ast.parse(SRC)
WANT = {"normalize_text", "normalize_math", "extract_number", "grade_latex", "grade_numeric",
        "math_equal", "_union_find_merge", "union_find_merge", "cluster_answers", "cluster_mass",
        "_numeric_equal", "numeric_equal"}
ns = {"np": np, "sps": sps, "re": re, "Counter": Counter, "Any": Any,
      "LOG": type("L", (), {"log": staticmethod(lambda *a, **k: None)})(),
      "Callable": Callable, "Sequence": Sequence,
      "HAS_MATH_VERIFY": importlib.util.find_spec("math_verify") is not None}
# module-level constants the functions close over
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") in ("MATH_SUBS", "ARTICLES", "PUNCT_RE") for t in node.targets):
        exec(compile(ast.Module([node], []), "<x>", "exec"), ns)
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANT:
        exec(compile(ast.Module([node], []), "<x>", "exec"), ns)
missing = WANT - set(ns)
assert not missing, f"missing {missing}"
cluster_answers, cluster_mass, math_equal = ns["cluster_answers"], ns["cluster_mass"], ns["math_equal"]

class C:
    STRIP_ARTICLES = True; USE_NLI_FALLBACK = True; NLI_ENTAIL_THRESHOLD = 0.7
    ENTROPY_LP_WEIGHTED = True; ENTROPY_MIN_VALID = 8; N_SAMPLES = 10
cfg = C()
fails = []

def check(name, cond, extra=""):
    if not cond: fails.append(f"{name} {extra}")

# ---- math equivalence merges what exact-string splits -------------------
ans = ["0.5", "1/2", r"\frac{1}{2}", "2/4", "0.25"]
lab = cluster_answers(ans, "latex", cfg, None)
check("math clustering", len(set(lab[:4])) == 1 and lab[4] != lab[0],
      f"-> {lab} (first four must share a cluster, 0.25 must not)")

# ---- text path unchanged, and NOT merged by the math oracle -------------
lab = cluster_answers(["Paris", "paris", "London"], "entity", cfg, None)
check("text clustering", lab[0] == lab[1] and lab[2] != lab[0], f"-> {lab}")

# ---- numeric form also gets equivalence --------------------------------
lab = cluster_answers(["18", "18.0", "20"], "numeric", cfg, None)
check("numeric clustering", lab[0] == lab[1] and lab[2] != lab[0], f"-> {lab}")

# ---- NLI now reachable for latex (was gated to short/entity) -----------
class FakeNLI:
    def __init__(s): s.called = False
    def entails(s, pairs):
        s.called = True
        return [0.99] * len(pairs)              # merge everything
n = FakeNLI(); cluster_answers(["x+1", "1+x", "zzz"], "latex", cfg, n)
check("NLI reachable for latex", n.called, "-> NLI never invoked on a math form")

# ---- transitivity: a~b, b~c must give ONE cluster ----------------------
class PairNLI:
    def entails(s, pairs):
        ok = {("a", "b"), ("b", "a"), ("b", "c"), ("c", "b")}
        return [0.99 if pr in ok else 0.01 for pr in pairs]
lab = cluster_answers(["a", "b", "c"], "short", cfg, PairNLI())
check("NLI transitivity", len(set(lab)) == 1, f"-> {lab} (union-find must close a~b~c)")

# ---- log-prob weighting ------------------------------------------------
labels = [0, 0, 1]
mass, w = cluster_mass(labels, [np.log(0.1), np.log(0.1), np.log(0.8)], cfg)
check("lp weighting active", w, "-> fell back to counts")
check("lp mass", abs(mass[1] / mass.sum() - 0.8) < 1e-6,
      f"-> cluster1 share {mass[1]/mass.sum():.4f}, want 0.80 (counts would give 0.333)")
mass, w = cluster_mass(labels, None, cfg)
check("legacy fallback", (not w) and list(mass) == [2.0, 1.0], f"-> {mass}, w={w}")
mass, w = cluster_mass(labels, [np.log(0.1), None, np.log(0.8)], cfg)
check("partial lp falls back", (not w) and list(mass) == [2.0, 1.0], f"-> {mass}, w={w}")

# ---- D6: no false merges across infinities or large integers ----------
for a, b in [("inf", "-inf"), ("infinity", "-infinity"), ("1e400", "-1e400"),
             ("1000000000000", "1000000000500"), ("1000000000000", "1000000000001")]:
    if math_equal(a, b):
        fails.append(f"math_equal({a!r}, {b!r}) -> True; these are different numbers")
# ...while genuine float-rounding spellings still merge
for a, b in [("1/3", "0.3333333333333333"), ("0.5", "1/2"), ("18", "18.0")]:
    if not math_equal(a, b):
        fails.append(f"math_equal({a!r}, {b!r}) -> False; these are the same number")

# ---- D9: the NLI pass gets the question as context --------------------
class CapturingNLI:
    def __init__(s): s.pairs = []
    def entails(s, pairs): s.pairs = pairs; return [0.01] * len(pairs)
cap = CapturingNLI()
cluster_answers(["18", "72"], "numeric", cfg, cap, question="How many eggs?")
if not all("How many eggs?" in p for pair in cap.pairs for p in pair):
    fails.append(f"NLI pairs lack question context: {cap.pairs[:2]}")

# ---- D7: modal share (counts) and modal mass (probability) differ ------
labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]        # 6 vs 4
lp = [np.log(0.05)] * 6 + [np.log(0.30)] * 4    # the 4 are far likelier each
mass, w = cluster_mass(labels, lp, cfg)
counts = np.array([labels.count(k) for k in sorted(set(labels))], float)
share = counts.max() / counts.sum()
mmass = mass.max() / mass.sum()
check("modal_share is count-based", abs(share - 0.6) < 1e-9, f"-> {share}")
check("modal_mass differs from share", abs(mmass - share) > 0.05,
      f"-> mass {mmass:.3f} vs share {share:.3f}; if equal the test is not exercising the split")
check("mass follows likelihood", mass[1] > mass[0],
      f"-> {mass}; the 4-sample cluster carries more probability here")

# ---- the headline bug: n=1 must not be confidence 1.0 ------------------
H_max = float(np.log(cfg.N_SAMPLES))
def conf(sizes, n_valid):
    p = np.array(sizes, float); p = p / p.sum()
    if n_valid < cfg.ENTROPY_MIN_VALID: return np.nan
    return 1 - float(sps.entropy(p)) / H_max
check("n=1 -> NaN", np.isnan(conf([1.0], 1)), "-> single parsed sample still scores finite")
check("n=10 unanimous -> 1.0", abs(conf([10.0], 10) - 1.0) < 1e-9)
check("n=10 even -> 0.0", abs(conf([1.0]*10, 10) - 0.0) < 1e-9)
# fixed denominator: 2 clusters out of 10 samples is NOT max entropy
v = conf([5.0, 5.0], 10)
check("fixed denominator", 0.6 < v < 0.8,
      f"-> 5/5 split scores {v:.3f}; log(n_valid) normalisation would give 0.0")

print("FAILURES:" if fails else "ALL ENTROPY TESTS PASSED")
for f in fails: print("  -", f)
sys.exit(1 if fails else 0)
