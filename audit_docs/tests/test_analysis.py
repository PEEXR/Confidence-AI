"""Integration test: H2/H3/GLMM/verbal/probe-C/gate3/manual-sheet on synthetic data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import NS, WRITES

import ast as _ast
_SRC = (Path(__file__).resolve().parents[2] / "confidence_pipeline.py").read_text()
_TREE = _ast.parse(_SRC)
def func_src(name):
    """Source of a top-level function, read from the file.

    `inspect.getsource` cannot see functions the harness exec'd from an AST.
    """
    for n in _TREE.body:
        if isinstance(n, (_ast.FunctionDef, _ast.ClassDef)) and n.name == name:
            return _ast.get_source_segment(_SRC, n) or ""
    raise AssertionError(f"{name} not found in confidence_pipeline.py")

def params_of(name):
    """Declared parameter names of a top-level function, from the AST."""
    for n in _TREE.body:
        if isinstance(n, _ast.FunctionDef) and n.name == name:
            a = n.args
            return [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    raise AssertionError(f"{name} not found in confidence_pipeline.py")
fails = []
def check(n, c, e=""):
    if not c: fails.append(f"{n} {e}")

class Cfg:
    SEED=7; QUADRANT_THRESHOLD=0.5; N_BOOTSTRAP=300; BOOTSTRAP_CI=0.95
    ECE_BINS=10; MURPHY_BINS=10; GATE1_AGREEMENT=0.95; HLR_METHOD="auto"
    PROBE_C_GRID=(0.01,0.1,1.0,10.0); PROBE_C_SELECT=True; PROBE_MAX_ITER=500
    AUROC_GATE=0.65; SURFACE_BASELINE=True; P0_NEG_CONTROL_TOL=0.10; GATE3_ENFORCE=True
cfg = Cfg()

# ---------------- synthetic signals -------------------------------------
rng = np.random.default_rng(0)
MODELS = ["qwen2.5-7b-base", "qwen2.5-7b-instruct"]
TIERS = ["R1", "R2", "C1"]
rows, bank = [], {t: [] for t in TIERS}
for t in TIERS:
    for i in range(160):
        qid = f"{t}-q{i}"
        bank[t].append({"qid": qid,
                        "question": ("What year did the very long and elaborate event happen in 1969 ?"
                                     if i % 2 else "Who?"),
                        "family": "retrieval" if t.startswith("R") else "reasoning",
                        "split": "test", "answer_form": "entity"})
for m in MODELS:
    for t in TIERS:
        for i in range(160):
            qid = f"{t}-q{i}"
            v = rng.uniform(0, 1); b = rng.uniform(0, 1)
            rows.append(dict(model=m, tier=t, qid=qid, split="test",
                             verbal_cal=v, behavioral_cal=b,
                             internal_cal=rng.uniform(0, 1),
                             correct=int(rng.random() < 0.5),
                             family="retrieval" if t.startswith("R") else "reasoning",
                             params_b=7.6, best_layer_pct=50.0,
                             delta_verbal_behavioral=v - b, in_band=(t != "R2")))
sig = pd.DataFrame(rows)

# ============ H2 ========================================================
h2 = NS["test_h2_quadrants"](sig, bank, cfg, gate1_pass=None, gate3={})
check("H2 gate1 conjunct present", h2["conjuncts"]["gate1_grading_sanity"] is None)
check("H2 blocked by null gate1", h2["h2_pass"] is False, f"-> {h2['verdict'][:60]}")
check("H2 names failing conjuncts", "gate1_grading_sanity" in h2["verdict"], f"-> {h2['verdict'][:90]}")
check("H2 labelled counts", set(h2["counts_labelled"]) <= set(NS["QUADRANT_LABELS"].values()),
      f"-> {list(h2['counts_labelled'])}")
check("H2 per-cell tests ran", any(h2["associations_per_cell"].values()),
      "-> no within-cell chi-square was computed")
check("H2 replication reported", "is_long" in h2["replication"])
rep = h2["replication"].get("is_long", {})
check("replication uses the conservative p", rep.get("p_used") == "p_conservative",
      f"-> {rep.get('p_used')}; beats_null must stay uncorrected to match its null, but a "
      f"claim that a feature GENERALISES needs the Yates p on small per-cell 2x2 tables")
# BEHAVIOURAL, not a label check. `p_used` reports the parameter; changing the
# lookup INSIDE the function left the label intact while the decision silently
# used the uncorrected p, and that flips `replicates`, which gates h2_pass.
_pc = {"c1": {"p": 0.04, "p_conservative": 0.10},
       "c2": {"p": 0.04, "p_conservative": 0.10}}
_cons = NS["replicates_across_cells"](_pc)
_unc = NS["replicates_across_cells"](_pc, p_key="p")
check("replication actually uses the conservative p",
      _cons["replicates"] is False and _unc["replicates"] is True,
      f"-> conservative={_cons['replicates']}, uncorrected={_unc['replicates']}; on a fixture "
      f"where p<0.05<p_conservative the two must disagree, or the lookup is not reading p_key")
check("per-cell tests carry both p values",
      all("p_conservative" in v for v in h2["associations_per_cell"].get("is_long", {}).values())
      if h2["associations_per_cell"].get("is_long") else True,
      "-> a per-cell association is missing p_conservative")
check("H2 delta reported", "delta_verbal_behavioral" in h2 and h2["delta_verbal_behavioral"]["n"] > 0)
check("H2 concentration reported", "hopeful" in h2.get("concentration", {}))
# even with everything else true, a False gate1 must still block
h2b = NS["test_h2_quadrants"](sig, bank, cfg, gate1_pass=False,
                              gate3={"c": {"passes": True}})
check("H2 blocked by failing gate1", h2b["h2_pass"] is False)
# threshold-degenerate cell detection
sig_deg = sig.copy(); sig_deg.loc[sig_deg.tier == "C1", "verbal_cal"] = 0.135
h2c = NS["test_h2_quadrants"](sig_deg, bank, cfg, gate1_pass=True, gate3={"c": {"passes": True}})
degen = [d["cell"] for d in h2c["threshold_degenerate_cells"]]
check("degenerate cells flagged", any("C1" in c for c in degen),
      f"-> {degen} (a cell whose verbal_cal never crosses 0.5 must be flagged)")

# ============ H3 — the NaN coercion bug ================================
h3sig = sig[sig.model.isin(MODELS)].copy()
# base model: verbal signal entirely missing, exactly the audit's scenario
h3sig.loc[h3sig.model == "qwen2.5-7b-base", "verbal_cal"] = np.nan
abst = {"per_cell": {}}
h3 = NS["test_h3_base_vs_instruct"](h3sig, abst, cfg)
check("H3 declines on all-NaN base", h3.get("available") is False,
      f"-> available={h3.get('available')}, rate_base={h3.get('hopeful_rate_base')}")
check("H3 reports no 0.0 rate", h3.get("hopeful_rate_base") in (None,),
      f"-> reported rate_base={h3.get('hopeful_rate_base')} (pre-fix code returned exactly 0.0)")
check("H3 counts excluded rows", h3.get("n_excluded_missing_signal", 0) > 0,
      f"-> {h3.get('n_excluded_missing_signal')}")
# partial missingness: must use only valid rows, not coerce
h3sig2 = sig.copy()
mask = (h3sig2.model == "qwen2.5-7b-base") & (h3sig2.index % 3 == 0)
h3sig2.loc[mask, "verbal_cal"] = np.nan
h32 = NS["test_h3_base_vs_instruct"](h3sig2, abst, cfg)
check("H3 available with partial NaN", h32.get("available") is True, f"-> {h32.get('verdict','')[:70]}")
if h32.get("available"):
    check("H3 excludes NaN rows", h32["n_excluded_missing_signal"] == int(mask.sum()),
          f"-> {h32['n_excluded_missing_signal']} vs {int(mask.sum())}")
    check("H3 separates correctness", "flagged_rows_correct_rate" in h32["per_model"][MODELS[0]])
    check("H3 confident_incorrect separate", "confident_incorrect_rate" in h32["per_model"][MODELS[0]])
    check("H3 neutral naming", "unwarranted_rate_base" in h32)
    check("H3 legacy alias kept", "hopeful_rate_base" in h32)

# ============ GLMM =====================================================
hlr = NS["hierarchical_regression"](sig, cfg, bank=bank)
check("GLMM available", hlr.get("available") is True, f"-> {hlr.get('reason')}")
req = hlr.get("random_intercepts_requested", [])
fit = hlr.get("random_intercepts_fitted", [])
check("GLMM requests 4 random intercepts", set(req) == {"question","model","tier","model_x_tier"},
      f"-> {req}")
# The honest-reporting property: a fit with NO random effects must not claim any.
check("fitted intercepts match the method",
      (hlr.get("method") == "bayes_mixed_glm") == bool(fit),
      f"-> method={hlr.get('method')} but random_intercepts_fitted={fit}; a cluster-robust "
      f"fallback has no random effects and must not report them")
check("non-convergence degrades", hlr.get("method") in
      ("bayes_mixed_glm", "logit_cluster_robust_cell"), f"-> {hlr.get('method')}")
if hlr.get("method") == "bayes_mixed_glm":
    check("converged flag set", hlr.get("converged") is True, f"-> {hlr.get('converged')}")
else:
    check("cluster unit is the cell", hlr.get("cluster_unit") == "model x tier cell",
          f"-> {hlr.get('cluster_unit')}; clustering on question would ignore cell baselines")
check("GLMM has question features", any(f in hlr.get("fixed_effects", []) for f in ("is_long","has_number")),
      f"-> {hlr.get('fixed_effects')}")
check("GLMM in_band covariate", "in_band" in hlr.get("fixed_effects", []), f"-> {hlr.get('fixed_effects')}")
check("GLMM per-cell fits", len(hlr.get("per_cell", {})) > 0, "-> no unpooled comparison")
check("GLMM did not fail", hlr.get("method") != "failed", f"-> {str(hlr.get('error'))[:120]}")
# Force the VB path to fail so the fallback runs, and check it reports honestly.
class CfgCR(Cfg): HLR_METHOD="cluster_robust"
hlr_cr = NS["hierarchical_regression"](sig, CfgCR(), bank=bank)
check("fallback method recorded", hlr_cr.get("method") == "logit_cluster_robust_cell",
      f"-> {hlr_cr.get('method')}")
check("fallback claims NO random intercepts", hlr_cr.get("random_intercepts_fitted") == [],
      f"-> {hlr_cr.get('random_intercepts_fitted')}; a cluster-robust logit has no random "
      f"effects and must not report any")
check("fallback still records what it wanted",
      set(hlr_cr.get("random_intercepts_requested", [])) ==
      {"question", "model", "tier", "model_x_tier"},
      f"-> {hlr_cr.get('random_intercepts_requested')}")
check("fallback clusters on the cell", hlr_cr.get("cluster_unit") == "model x tier cell",
      f"-> {hlr_cr.get('cluster_unit')}")

check("GLMM records what it fitted",
      bool(hlr.get("random_intercepts_fitted")) or bool(hlr.get("cluster_unit")),
      f"-> method={hlr.get('method')}")

# ============ verbal axis ==============================================
g = []
for m in MODELS:
    for t in TIERS:
        for i in range(120):
            qid=f"{t}-q{i}"; c=int(rng.random()<0.5)
            for var, extra in (("A", dict(confidence=rng.uniform(0,1), bucket=None, decision=None)),
                               ("B", dict(confidence=None, bucket="CERTAIN" if c else "NO_IDEA", decision=None)),
                               ("C", dict(confidence=None, bucket=None,
                                          decision="ANSWER" if rng.random()<0.5 else "PASS"))):
                g.append(dict(model=m, tier=t, qid=qid, split="calibration", variant=var,
                              sample_idx=0, is_agreement=False, correct=c, **extra))
graded = pd.DataFrame(g)
bmap = NS["empirical_bucket_map"](graded, cfg)
vs = NS["verbal_scores"](graded, bmap, cfg)
check("Bfix variant emitted", "Bfix" in set(vs.variant), f"-> {sorted(set(vs.variant))}")
bf = vs[vs.variant=="Bfix"]
check("Bfix uses fixed values", set(bf.verbal_raw.dropna()) <= set(NS["BUCKET_FIXED_VALUES"].values()),
      f"-> {sorted(set(bf.verbal_raw.dropna()))}")
# B is perfectly label-derived here -> lowest Brier -> pre-fix code would pick it
fmt = {}
for v in NS["VERBAL_FORMATS"]:
    s = vs[(vs.variant==v)&(vs.split=="calibration")].dropna(subset=["verbal_raw"])
    if len(s) >= 20:
        fmt[v] = NS["brier"](s.verbal_raw.values, s.correct.values.astype(float))
check("B has the lowest Brier here", fmt.get("B") == min(fmt.values()),
      f"-> {({k:round(v,4) for k,v in fmt.items()})} (test is only meaningful if B wins)")
elig = [v for v in NS["LABEL_FREE_FORMATS"] if v in fmt]
canonical = min(elig, key=lambda v: fmt[v])
check("B never canonical", canonical != "B", f"-> chose {canonical}")
check("canonical is label-free", canonical in NS["LABEL_FREE_FORMATS"], f"-> {canonical}")

# ============ probe C selection ========================================
X = rng.normal(size=(200, 30)); w = rng.normal(size=30)
y = (X @ w + rng.normal(scale=2.0, size=200) > 0).astype(int)
C_sel, meta = NS["select_probe_C"](X[:120], y[:120], cfg, 0)
check("C selected from grid", C_sel in cfg.PROBE_C_GRID, f"-> {C_sel}")
check("C picked by train CV", meta["method"] == "train_cv_auroc", f"-> {meta}")
check("C grid recorded", len(meta["grid"]) == len(cfg.PROBE_C_GRID), f"-> {meta['grid']}")
check("C used >1 fold", meta["n_folds"] >= 2, f"-> {meta['n_folds']}")
class Cfg2(Cfg): PROBE_C_SELECT=False
C2, m2 = NS["select_probe_C"](X[:120], y[:120], Cfg2(), 0)
check("C selection disableable", C2 == 1.0 and m2["method"] == "fixed", f"-> {C2}, {m2['method']}")
# selection must not touch the calibration split — that split gates the probe,
# so choosing C to maximise its AUROC would be selection on the test statistic
check("C selection sees no calibration data",
      not ({"X_ca", "y_ca"} & set(params_of("select_probe_C"))),
      f"-> select_probe_C params {params_of('select_probe_C')} include calibration arrays")
# degenerate input must not crash
Cd, md = NS["select_probe_C"](X[:20], np.zeros(20, int), cfg, 0)
check("single-class train handled", Cd == 1.0 and md["n_folds"] == 0, f"-> {Cd}, {md['method']}")

# ============ gate3 p0 negative control ================================
def sweep_row(pct, auroc, **kw):
    d = dict(cell="m__t", model="m", tier="t", layer_pct=pct, auroc_cal=auroc,
             nonfinite_frac=0.0,
             meets_gate=auroc>=0.65, beats_null=True, beats_surface=True, probe_C=1.0)
    d.update(kw); return d
# the audit's exact case: p0 = 0.717 on a token that must be constant
sw = pd.DataFrame([sweep_row(0,0.717), sweep_row(50,0.80)])
g3 = NS["gate3_verdict"](sw, cfg)
check("p0=0.717 fails gate3", g3["m__t"]["passes"] is False, f"-> {g3['m__t']}")
check("p0 failure diagnosed", "p0 negative control FAILED" in g3["m__t"]["diagnosis"],
      f"-> {g3['m__t']['diagnosis']}")
sw2 = pd.DataFrame([sweep_row(0,0.52), sweep_row(50,0.80)])
g3b = NS["gate3_verdict"](sw2, cfg)
check("clean p0 passes", g3b["m__t"]["passes"] is True, f"-> {g3b['m__t']['diagnosis']}")
sw3 = pd.DataFrame([sweep_row(50,0.80)])          # no p0 at all
g3c = NS["gate3_verdict"](sw3, cfg)
check("missing p0 does NOT pass", g3c["m__t"]["passes"] is False, f"-> {g3c['m__t']['diagnosis']}")
sw4 = pd.DataFrame([sweep_row(0,0.52,nonfinite_frac=0.01), sweep_row(50,0.80)])
g3d = NS["gate3_verdict"](sw4, cfg)
check("dirty activations fail", g3d["m__t"]["passes"] is False, f"-> {g3d['m__t']['diagnosis']}")
g3e = NS["gate3_verdict"](pd.DataFrame(), cfg, blocked={"x__y": {"stored_nonfinite_frac": 0.3}})
check("blocked cells appear in gate3", g3e.get("x__y", {}).get("passes") is False, f"-> {g3e}")

# ============ Gate 3 must be enforced on the DATA path =================
sw_bad = pd.DataFrame([sweep_row(0, 0.717, cell="m__t", model="m", tier="t"),
                       sweep_row(50, 0.80, cell="m__t", model="m", tier="t")])
g3bad = NS["gate3_verdict"](sw_bad, cfg)
check("gate3 marks the p0 cell failed", g3bad["m__t"]["passes"] is False)
check("internal_scores takes gate3", "gate3" in params_of("internal_scores"),
      "-> a cell that fails Gate 3 can still emit internal_cal downstream")
check("assemble_signals takes gate3", "gate3" in params_of("assemble_signals"),
      "-> gate3 never reaches the signal table")
# BEHAVIOURAL: a failing cell must emit zero rows, and a passing one must not.
def _fake_sweep(cell, model, tier, auroc):
    return [dict(cell=cell, model=model, tier=tier, layer_pct=pc, auroc_cal=auroc,
                 nonfinite_frac=0.0, meets_gate=True, beats_null=True, beats_surface=True,
                 probe_C=1.0) for pc in (0, 50)]
class _Cfg3(Cfg): GATE3_ENFORCE=True; PROBE_LABEL="correct"

# The emit path needs real activations and labels, or the function returns 0
# rows for unrelated reasons and the test passes without testing anything.
_NQ = 120
_qids = [f"t-q{i}" for i in range(_NQ)]
_rg3 = np.random.default_rng(4)
_X = _rg3.normal(size=(_NQ, 12))
_bank3 = {"t": [{"qid": q, "split": ("train" if i < 80 else "test"), "question": "q",
                 "family": "retrieval"} for i, q in enumerate(_qids)]}
_graded3 = pd.DataFrame([dict(model="m", tier="t", qid=q, variant="EXTRACT", sample_idx=0,
                              correct=int(_X[i, 0] > 0)) for i, q in enumerate(_qids)])
def _run_internal(gate3):
    saved = NS.get("load_activations")
    NS["load_activations"] = lambda model, tier: (_qids, {0: _X, 50: _X})
    try:
        return NS["internal_scores"](pd.DataFrame(_fake_sweep("m__t", "m", "t", 0.8)),
                                     _bank3, _graded3, pd.DataFrame(), {}, _Cfg3(),
                                     gate3=gate3)
    finally:
        if saved is not None: NS["load_activations"] = saved

_baseline = _run_internal({"m__t": {"passes": True}})
check("the gate3 test can actually emit rows", len(_baseline) > 0,
      "-> a passing cell emitted 0 rows, so the skip test below proves nothing")
isr = _run_internal({"m__t": {"passes": False, "diagnosis": "p0 failed"}})
check("failing cell emits no internal rows", len(isr) == 0,
      f"-> {len(isr)} rows (a passing cell emits {len(_baseline)})")
isr2 = _run_internal({"other__cell": {"passes": True}})
check("cell with no verdict is skipped", len(isr2) == 0,
      f"-> {len(isr2)} rows; a missing verdict must not read as a pass")
try:
    _run_internal(None)
    fails.append("internal_scores with gate3=None did not raise — enforcement fails open")
except RuntimeError:
    pass

# H2's gate3 conjunct must be per-cell, not grid-wide
h2g = NS["test_h2_quadrants"](sig, bank, cfg, gate1_pass=True,
                              gate3={"qwen2.5-7b-base__R1": {"passes": True}})
check("gate3 conjunct is per-cell", h2g["conjuncts"]["gate3_probe_validity"] is False,
      f"-> one passing cell out of {len(h2g.get('gate3_cells_used', []))} satisfied the conjunct "
      f"for the whole grid")
check("failing cells named", len(h2g.get("gate3_cells_failing", [])) > 0,
      f"-> {h2g.get('gate3_cells_failing')}")
allpass = {c: {"passes": True} for c in
           sorted({f"{m}__{t}" for m in MODELS for t in TIERS})}
h2p = NS["test_h2_quadrants"](sig, bank, cfg, gate1_pass=True, gate3=allpass)
check("all-pass satisfies conjunct", h2p["conjuncts"]["gate3_probe_validity"] is True,
      f"-> {h2p.get('gate3_cells_failing')}")

# ============ ground-truth merge: conflicts must be visible ============
_logs = []
_realLOG = NS["LOG"]
class _SpyLOG:
    def log(self, ev, **kw): _logs.append((ev, kw))
NS["LOG"] = _SpyLOG()
try:
    g_ok = pd.DataFrame([dict(model="m", tier="t", qid=f"q{i}", variant="EXTRACT",
                              sample_idx=0, correct=i % 2, bucket=None, decision=None,
                              confidence=None, split="test", is_agreement=False)
                         for i in range(40)])
    # same key graded two different ways
    g_bad = pd.concat([g_ok, g_ok.head(5).assign(correct=lambda d: 1 - d["correct"])],
                      ignore_index=True)
    dup = (g_bad.groupby(["model", "tier", "qid"])["correct"].nunique().pipe(lambda x: x[x > 1]))
    check("conflict fixture really conflicts", len(dup) == 5, f"-> {len(dup)}")
    _logs.clear()
    conflicts = (g_bad.groupby(["model", "tier", "qid"])["correct"].nunique()
                 .pipe(lambda x: x[x > 1]))
    src_gt = func_src("assemble_signals")
    check("assemble_signals detects conflicts before dedup",
          "ground_truth_conflicts" in src_gt and
          src_gt.index("ground_truth_conflicts") < src_gt.index("drop_duplicates"),
          "-> conflicts are dropped silently, turning a data-integrity clash into a coin flip")
    check("conflicts are logged, not just dropped",
          "ground_truth_duplicates" in src_gt, "-> duplicate count not logged")
finally:
    NS["LOG"] = _realLOG

# ============ Format A parses the model's stated number ================
pn = NS["parse_confidence_numeric"]
PARSE_CASES = [
    ("90", 0.90), ("100", 1.0), ("0", 0.0), ("85.5", 0.855), ("CONFIDENCE: 75", 0.75),
    ("85%", 0.85), ("100%", 1.0), ("0.5%", 0.005), ("95 %", 0.95),
    ("0.9", 0.9), (".85", 0.85), ("1.0", 1.0), ("1", 0.01),
    ("7 out of 10", 0.7), ("7/10", 0.7), ("1/2", 0.5), ("1 in 10", 0.1),
    ("10 out of 10", 1.0), ("about 7 out of 10", 0.7),
    # "50/50" is the English idiom for maximum uncertainty. Reading it as
    # 50/50 = 1.0 would record perfect confidence for a model saying it had
    # none — manufacturing the exact phenomenon this project measures.
    ("50/50", 0.5), ("50-50", 0.5), ("fifty-fifty", 0.5), ("it's 50:50", 0.5),
    ("85/100", 0.85),
    # A ratio the parser cannot read must be None, not the numerator on a
    # 0-100 scale: falling through makes "21 out of 21" score 0.21 against
    # 1.0 for "20 out of 20" — a 5x cliff either side of the bound, and wrong
    # by exactly den/100. An unparsed row is visible in the compliance rate.
    ("20/20", 1.0), ("20 out of 20", 1.0),
    ("21/21", None), ("21 out of 21", None), ("40/50", None), ("17/21", None),
    ("1/1000", None),
    # A stated RANGE must not be read as a negative number. Capturing the sign
    # made "90-95%" match "-95", which the percentage branch then rejected.
    ("90-95%", 0.95), ("80-90%", 0.90), ("between 80-90%", 0.90), ("90 - 95%", 0.95),
    ("-5", None), ("-0.9", None), ("150", None), ("abc", None),
]
for raw, want in PARSE_CASES:
    got = pn(raw)
    ok = (got is None and want is None) or (
        got is not None and want is not None and abs(got - want) < 1e-9)
    check(f"parse {raw!r}", ok, f"-> {got}, want {want}")

# ============ manual sheet ingest ======================================
out = Path(str(Path(__file__).resolve().parent / "_out"))
out.mkdir(parents=True, exist_ok=True)
sheet = out / "sheet.csv"
pd.DataFrame({"automated_correct":[True]*10+[False]*10,
              "manual_correct":["yes"]*9+["no"]+[0]*10,
              "grader":["symbolic"]*20,
              "disagreement_note":[""]*9+["latex formatting"]+[""]*10}).to_csv(sheet, index=False)
man = NS["ingest_manual_sheet"](sheet, cfg)
check("sheet ingested", man["available"] is True, f"-> {man}")
check("agreement correct", abs(man["agreement"]-0.95) < 1e-9, f"-> {man['agreement']}")
check("disagreements counted", man["n_disagreements"] == 1, f"-> {man['n_disagreements']}")
check("notes captured", man["disagreement_notes"] == ["latex formatting"], f"-> {man['disagreement_notes']}")
blank = out / "blank.csv"
pd.DataFrame({"automated_correct":[True]*5, "manual_correct":[""]*5}).to_csv(blank, index=False)
mb = NS["ingest_manual_sheet"](blank, cfg)
check("blank sheet not available", mb["available"] is False and mb["n_labelled"] == 0, f"-> {mb}")
check("blank gate1 is None", NS["combined_gate1"]({}, mb, cfg) is None,
      "-> an unfilled sheet must not read as pass or fail")
check("filled sheet drives gate1", NS["combined_gate1"]({"gate1_pass": False}, man, cfg) is True,
      "-> human sheet must override the automated judge")

# ============ chi2 fast path must equal scipy, 2x2 included ============
import scipy.stats as _sps
import pandas as _pd
def _bincount_chi2(qc, fc):
    R, C = int(qc.max())+1, int(fc.max())+1
    obs = np.bincount(qc*C+fc, minlength=R*C).reshape(R, C).astype(float)
    rt, ct, n = obs.sum(1), obs.sum(0), obs.sum()
    exp = np.outer(rt, ct)/n
    nz = exp > 0
    return float((((obs-exp)**2)[nz]/exp[nz]).sum())
rg = np.random.default_rng(3)
worst = 0.0
shapes_2x2 = 0
for _ in range(200):
    R, C = int(rg.integers(2, 5)), int(rg.integers(2, 5))
    n = int(rg.integers(60, 400))
    qc = rg.integers(0, R, n); fc = rg.integers(0, C, n)
    if len(set(qc)) < R or len(set(fc)) < C: continue
    obs = np.zeros((R, C))
    for a, b in zip(qc, fc): obs[a, b] += 1
    if (obs.sum(0) == 0).any() or (obs.sum(1) == 0).any(): continue
    if R == 2 and C == 2: shapes_2x2 += 1
    # the pipeline's observed statistic must be the UNCORRECTED one, so it is
    # comparable with the permutation null
    ref = _sps.chi2_contingency(obs, correction=False)[0]
    worst = max(worst, abs(_bincount_chi2(qc, fc) - ref))
check("chi2 fast path == scipy(correction=False)", worst < 1e-6, f"-> max diff {worst:.2e}")
check("2x2 tables were exercised", shapes_2x2 >= 10, f"-> only {shapes_2x2} of 200")
# and the pipeline must not be using the Yates-corrected value
# AST, not text: a `correction=False` mention in a COMMENT satisfied the old
# string check, so reverting the actual call still passed.
def _chi2_calls(fn_name):
    for n in _TREE.body:
        if isinstance(n, _ast.FunctionDef) and n.name == fn_name:
            return [c for c in _ast.walk(n) if isinstance(c, _ast.Call)
                    and _ast.unparse(c.func).endswith("chi2_contingency")]
    return []
calls = _chi2_calls("test_h2_quadrants")
check("chi2_contingency calls found", len(calls) >= 1, f"-> {len(calls)}")
kwsets = [{k.arg: _ast.unparse(k.value) for k in c.keywords} for c in calls]
check("observed chi2 disables Yates",
      any(k.get("correction") == "False" for k in kwsets),
      f"-> keywords {kwsets}; scipy corrects 2x2 by default while the permutation null "
      f"does not, biasing beats_null toward False")
check("a conservative p is also computed",
      any("correction" not in k for k in kwsets),
      f"-> {kwsets}; replication needs the Yates p on small 2x2 tables")
# A 2x2 fixture: exactly two quadrant levels, so Yates would apply. Without
# one, every association in the default fixture is 4x2 and the corrected and
# uncorrected statistics are identical — the regression would be invisible.
rows2 = []
rg2 = np.random.default_rng(5)
for i in range(200):
    qid = f"R1-q{i}"
    long_q = i % 2 == 0
    # verbal always high; behavioral high iff long -> only agree_high/hopeful
    v = 0.9
    b = 0.9 if (long_q and rg2.random() < 0.75) else 0.1
    rows2.append(dict(model="m2", tier="R1", qid=qid, split="test", verbal_cal=v,
                      behavioral_cal=b, internal_cal=b, correct=int(rg2.random() < 0.5),
                      family="retrieval", params_b=7.6, best_layer_pct=50.0,
                      delta_verbal_behavioral=v - b, in_band=True))
bank2 = {"R1": [{"qid": f"R1-q{i}",
                 # is_long is len(split()) > 15, so the long variant must
                 # actually clear sixteen words.
                 "question": (("a very long and quite elaborate question with a great many "
                               "words in it indeed for the purposes of this test")
                              if i % 2 == 0 else "Who?"),
                 "family": "retrieval", "split": "test", "answer_form": "entity"}
                for i in range(200)]}
h22 = NS["test_h2_quadrants"](pd.DataFrame(rows2), bank2, cfg, gate1_pass=True,
                              gate3={"m2__R1": {"passes": True}})
a2 = h22.get("associations", {}).get("is_long")
check("2x2 fixture produced an association", a2 is not None,
      f"-> {list(h22.get('associations', {}))}")
if a2:
    tab2 = _pd.DataFrame(a2["table"]).values.astype(float)
    check("fixture really is 2x2", tab2.shape == (2, 2), f"-> {tab2.shape}")
    unc = _sps.chi2_contingency(tab2, correction=False)[0]
    cor = _sps.chi2_contingency(tab2)[0]
    check("reported chi2 is the UNCORRECTED one", abs(a2["chi2"] - unc) < 1e-9,
          f"-> reported {a2['chi2']:.6f}, uncorrected {unc:.6f}, Yates {cor:.6f}")
    check("Yates would have differed here", abs(unc - cor) > 1e-6,
          f"-> unc {unc:.6f} == cor {cor:.6f}; fixture cannot detect the regression")
    check("conservative p is the Yates one",
          abs(a2["p_conservative"] - _sps.chi2_contingency(tab2)[1]) < 1e-9,
          f"-> {a2.get('p_conservative')}")
    # The NULL must be uncorrected too — testing the pipeline's own null, not
    # a reimplementation. The permutation p95 on a 2x2 sits on a lumpy discrete
    # distribution and moves between adjacent atoms, so the MEAN is the stable
    # discriminator: ~dof (=1 here) uncorrected, materially lower with Yates.
    check("null_mean reported", np.isfinite(a2.get("null_mean", np.nan)),
          f"-> {a2.get('null_mean')}")
    check("permutation null is uncorrected", 0.90 <= a2["null_mean"] <= 1.15,
          f"-> null_mean {a2['null_mean']:.4f}; uncorrected is ~1.01 for this table and "
          f"Yates-corrected is ~0.80, so observed and null are on different conventions")

h2y = NS["test_h2_quadrants"](sig, bank, cfg, gate1_pass=True, gate3={"c": {"passes": True}})
assoc = h2y.get("associations", {})
check("yates flag recorded", all(v.get("yates_correction") is False for v in assoc.values()),
      f"-> {[(k, v.get('yates_correction')) for k, v in assoc.items()]}")

# nulls must not depend on row order or on how many tests ran before
sig_shuf = sig.sample(frac=1.0, random_state=99).reset_index(drop=True)
h2a = NS["test_h2_quadrants"](sig, bank, cfg, gate1_pass=True, gate3={"c": {"passes": True}})
h2b = NS["test_h2_quadrants"](sig_shuf, bank, cfg, gate1_pass=True, gate3={"c": {"passes": True}})
for f in h2a["associations"]:
    a, b = h2a["associations"][f], h2b["associations"].get(f, {})
    if abs(a["null_p95"] - b.get("null_p95", np.nan)) > 1e-9:
        fails.append(f"null_p95 for {f} depends on row order: {a['null_p95']} vs {b.get('null_p95')}")

# ============ D10: MNAR NaN must not be silently absorbed ==============
sig_nan = sig.copy()
# ENTROPY_MIN_VALID drops individual QUESTIONS, concentrated in the hard cells
# but not identical to them — if missingness lined up exactly with a tier, the
# tier term would absorb it and the indicator would be collinear by
# construction rather than by data.
_hardish = (sig_nan.tier == "C1") & (sig_nan.index % 4 != 0)
sig_nan.loc[_hardish, "behavioral_cal"] = np.nan
h2n = NS["test_h2_quadrants"](sig_nan, bank, cfg, gate1_pass=True, gate3={"c": {"passes": True}})
check("h2 records other_cal composition", "other_cal_rows_one_component" in h2n,
      "-> a row where behavioral is missing silently becomes internal-only")
check("h2 restricts to complete rows", h2n["other_cal_rows_one_component"] == 0
      or h2n.get("n", 0) > 0, f"-> {h2n.get('other_cal_rows_one_component')}")
hlrn = NS["hierarchical_regression"](sig_nan, cfg, bank=bank)
check("GLMM counts imputed rows", hlrn.get("imputed_counts", {}).get("behavioral_cal", 0) > 0,
      f"-> {hlrn.get('imputed_counts')}")
_dropped = hlrn.get("terms_dropped_degenerate", {})
check("GLMM adds missingness indicator",
      "behavioral_cal_missing" in hlrn.get("fixed_effects", [])
      or "behavioral_cal_missing" in _dropped,
      f"-> fixed={hlrn.get('fixed_effects')}, dropped={_dropped}; mean-imputing MNAR values "
      f"without an indicator biases the behavioral coefficient toward zero, and if the "
      f"indicator is collinear that has to be RECORDED, not silent")
check("indicator is used, not silently dropped",
      "behavioral_cal_missing" in hlrn.get("fixed_effects", []),
      f"-> dropped as {_dropped.get('behavioral_cal_missing')}; the fixture should make "
      f"missingness partial within the cell so the indicator is estimable")

# ============ H1 must declare when its signals cover different cells =====
sig_part = sig.copy()
sig_part.loc[sig_part.tier == "C1", "internal_cal"] = np.nan     # Gate 3 dropped this cell
h1p = NS["test_h1_signal_calibration"](sig_part, cfg)
cov = h1p.get("coverage", {})
check("H1 reports coverage", bool(cov), "-> no coverage block")
check("H1 flags a non-like-for-like comparison", cov.get("like_for_like") is False,
      f"-> like_for_like={cov.get('like_for_like')}; internal covers "
      f"{cov.get('per_signal', {}).get('internal', {}).get('n_cells')} cells vs "
      f"{cov.get('cells_total')} total, so pooled ECE/Brier and the best-worst delta are "
      f"computed over different question sets")
check("H1 names the missing cells",
      len(cov.get("per_signal", {}).get("internal", {}).get("missing_cells", [])) > 0,
      f"-> {cov.get('per_signal', {}).get('internal', {})}")
check("H1 carries a warning string", bool(cov.get("warning")), "-> no warning text")
h1f = NS["test_h1_signal_calibration"](sig, cfg)
check("full coverage is like-for-like", h1f["coverage"]["like_for_like"] is True,
      f"-> {h1f['coverage'].get('warning')}")

# ============ H4 must not draw depth curves from failed cells ===========
sweep_h4 = pd.DataFrame([
    dict(cell=f"{m}__{t}", model=m, tier=t, layer_pct=pc,
         auroc_cal=(0.80 if pc >= 50 else 0.52), family=("retrieval" if t.startswith("R") else "reasoning"),
         params_b=7.6, nonfinite_frac=0.0, meets_gate=(pc >= 50), beats_null=True,
         beats_surface=True, probe_C=1.0)
    for m in MODELS for t in TIERS for pc in (0, 25, 50, 75, 100)])
g3_mixed = {f"{m}__{t}": {"passes": (t != "C1")} for m in MODELS for t in TIERS}
h4g = NS["test_h4_depth"](sweep_h4, cfg, gate3=g3_mixed)
check("H4 excludes gate3 failures",
      sorted(h4g.get("gate3_excluded_cells", [])) == sorted(f"{m}__C1" for m in MODELS),
      f"-> {h4g.get('gate3_excluded_cells')}; the depth curves are the repo's best finding "
      f"and must not include cells whose p0 control failed")
check("H4 reports its shrunk grid", h4g.get("coverage_note") is not None,
      "-> no coverage note on a reduced grid")
h4all = NS["test_h4_depth"](sweep_h4, cfg,
                            gate3={f"{m}__{t}": {"passes": True} for m in MODELS for t in TIERS})
check("H4 keeps every passing cell", h4all.get("gate3_excluded_cells") == [],
      f"-> {h4all.get('gate3_excluded_cells')}")
check("H4 grid shrank", h4g.get("n_cells", 0) < h4all.get("n_cells", 0),
      f"-> {h4g.get('n_cells')} vs {h4all.get('n_cells')}")
# Fail CLOSED. Returning the unfiltered sweep on a missing verdict meant one
# dropped keyword argument reverted H4's gating silently, and an empty
# gate3_excluded_cells then reads as "everything passed".
for bad_gate in (None, {}):
    try:
        NS["gate3_filter_sweep"](sweep_h4, bad_gate, cfg)
        fails.append(f"gate3_filter_sweep({bad_gate!r}) returned instead of raising — "
                     f"H4 enforcement fails open")
    except RuntimeError:
        pass
    try:
        NS["test_h4_depth"](sweep_h4, cfg, gate3=bad_gate)
        fails.append(f"test_h4_depth(gate3={bad_gate!r}) did not raise — "
                     f"dropping the kwarg silently reverts the fix")
    except RuntimeError:
        pass
class CfgOff(Cfg): GATE3_ENFORCE=False
off, exc_off = NS["gate3_filter_sweep"](sweep_h4, None, CfgOff())
check("GATE3_ENFORCE=False is an explicit opt-out",
      len(off) == len(sweep_h4) and exc_off == [], f"-> {len(off)}, {exc_off}")

# ============ an entirely-missing signal must not kill the GLMM ==========
# Reachable, not hypothetical: while the tap defect persists the p0 control
# fails grid-wide, every cell is skipped, and internal_cal is 100% NaN.
sig_none = sig.copy(); sig_none["internal_cal"] = np.nan
hlrn2 = NS["hierarchical_regression"](sig_none, cfg, bank=bank)
check("GLMM survives an all-NaN signal", hlrn2.get("method") != "failed",
      f"-> {hlrn2.get('method')}: {str(hlrn2.get('error'))[:120]}")
check("GLMM names the dropped signal",
      "internal_cal" in hlrn2.get("signals_dropped_all_missing", []),
      f"-> {hlrn2.get('signals_dropped_all_missing')}")
check("dropped signal leaves the formula", "internal_cal" not in hlrn2.get("fixed_effects", []),
      f"-> {hlrn2.get('fixed_effects')}")
check("its interactions leave too", "internal_cal" not in hlrn2.get("formula", ""),
      f"-> {hlrn2.get('formula')}")
check("GLMM still fitted rows", hlrn2.get("n", 0) > 0, f"-> n={hlrn2.get('n')}")
# `layer_pct` is an internal-probe artefact. When the probe is gone,
# `best_layer_pct` is gone with it and filling 50.0 makes layer_pct a
# constant, so `layer_pct:is_reasoning` is a scalar multiple of `is_reasoning`
# and the design matrix is singular.
sig_none2 = sig.copy()
sig_none2["internal_cal"] = np.nan
sig_none2["best_layer_pct"] = np.nan
hlrn3 = NS["hierarchical_regression"](sig_none2, cfg, bank=bank)
check("GLMM survives internal+layer_pct both gone", hlrn3.get("method") != "failed",
      f"-> {hlrn3.get('method')}: {str(hlrn3.get('error'))[:140]}")
check("layer_pct interaction leaves too", "layer_pct" not in hlrn3.get("formula", ""),
      f"-> {hlrn3.get('formula')}")
# The per-cell fits must use the same surviving signals as the pooled fit;
# hardcoding all three killed every one of them with the identical error.
pc3 = hlrn3.get("per_cell", {})
check("per-cell fits ran", len(pc3) > 0, "-> no per-cell fits at all")
check("per-cell fits did not all error",
      sum(1 for v in pc3.values() if "error" in v) < len(pc3),
      f"-> {sum(1 for v in pc3.values() if 'error' in v)}/{len(pc3)} errored; the per-cell "
      f"formula is not tracking the surviving signals")
check("degenerate terms are recorded", isinstance(hlrn3.get("terms_dropped_degenerate"), dict),
      f"-> {hlrn3.get('terms_dropped_degenerate')}")

# ============ probe_labels: the entropy median must not cross splits ====
ent = pd.DataFrame({
    "model": ["m"]*10, "tier": ["t"]*10, "qid": [f"q{i}" for i in range(10)],
    # train rows are low, test rows are high: a pooled median differs from a train median
    "confidence_behavioral": [0.1,0.1,0.1,0.2,0.2, 0.9,0.9,0.9,0.9,0.9],
})
qids = [f"q{i}" for i in range(10)]
split_of = {q: ("train" if i < 5 else "test") for i, q in enumerate(qids)}
class CfgE(Cfg): PROBE_LABEL="entropy"
y_train_med, _ = NS["probe_labels"]("m","t",qids,pd.DataFrame(),ent,CfgE(),split_of=split_of)
y_pooled_med, _ = NS["probe_labels"]("m","t",qids,pd.DataFrame(),ent,CfgE(),split_of=None)
check("entropy median uses train only", list(y_train_med) != list(y_pooled_med),
      f"-> train-median labels {list(y_train_med)} equal pooled-median labels; "
      f"the test split is influencing its own labels")
check("train-median labels sane", y_train_med[5:].sum() == 5,
      f"-> {list(y_train_med)}; all test rows are above the train median here")

# ============ popularity + difficulty matching =========================
pop = NS["popularity_contrast"](sig, graded, cfg)
check("popularity available", pop["available"] is True, f"-> {pop.get('reason')}")
check("R1/R2 contrast computed", all("accuracy_drop_R1_minus_R2" in v for v in pop["contrasts"].values()))
check("R3 coverage warned", "coverage_warning" in pop, "-> R3 absent but not flagged")
dm = NS["difficulty_matched_view"](sig, cfg)
check("difficulty matching available", dm["available"] is True, f"-> {dm}")
check("difficulty bins present", any(v["bins"] for v in dm["per_tier"].values()))
first = next(iter(dm["per_tier"].values()))
b0 = next(iter(first["bins"].values()))
check("resolution reported per bin", any("resolution" in s for s in b0["signals"].values()),
      f"-> {b0['signals']}")

# ======================================================================
# END-TO-END: assemble_signals, the function every other check assumed.
#
# Verification found the canonical-format test re-implementing the selection
# inline instead of calling the real code, and the ground-truth conflict path
# checked only by grepping the source. Both are exercised for real here.
# ======================================================================
_MODELS2, _TIERS2 = ["mA", "mB"], ["R1"]
NS["MODEL_SPECS"] = {m: {"params_b": 7.6, "layers": 28} for m in _MODELS2}
NS["TIER_SPECS"] = {"R1": {"family": "retrieval", "answer_form": "entity"}}
_rgE = np.random.default_rng(11)
_QN = 300
_bankE = {"R1": [{"qid": f"R1-q{i}",
                  "split": ("train" if i < 150 else "calibration" if i < 225 else "test"),
                  "question": "Who wrote it?", "family": "retrieval"} for i in range(_QN)]}
_split = {r["qid"]: r["split"] for r in _bankE["R1"]}

# Four buckets with GRADED accuracy. Two buckets, or four that split cleanly
# into "right" and "wrong", both collapse B's empirical mapping to two distinct
# values — B is then excluded by the §8·6 distinctness pre-flight and the test
# would pass for the wrong reason. It has to be excluded for being
# LABEL-DERIVED, which means it must first be a fully eligible candidate.
_BUCKET_RATE = {"CERTAIN": 0.95, "FAIRLY_CONFIDENT": 0.75,
                "SOMEWHAT_UNSURE": 0.45, "NO_IDEA": 0.10}

def _graded_rows(dup_conflict=False):
    out = []
    for m in _MODELS2:
        for i in range(_QN):
            qid = f"R1-q{i}"
            _b = list(_BUCKET_RATE)[int(_rgE.integers(0, 4))]
            corr = int(_rgE.random() < _BUCKET_RATE[_b])
            base = dict(model=m, tier="R1", qid=qid, split=_split[qid], sample_idx=0,
                        is_agreement=(i < 60), correct=corr, answer_form="entity",
                        gold="x", answer="x", grader="string")
            # A: a genuine stated number, uncorrelated with correctness
            out.append({**base, "variant": "A", "confidence": float(_rgE.integers(0, 101)) / 100,
                        "bucket": None, "decision": None})
            # B: the bucket the model chose, whose mapped value IS the empirical
            # accuracy of that bucket — so B tracks correctness by construction
            # and wins any Brier-based selection.
            out.append({**base, "variant": "B", "confidence": None,
                        "bucket": _b, "decision": None})
            out.append({**base, "variant": "C", "confidence": None, "bucket": None,
                        "decision": ("ANSWER" if _rgE.random() < 0.5 else "PASS")})
            out.append({**base, "variant": "EXTRACT", "confidence": None, "bucket": None,
                        "decision": None})
    if dup_conflict:
        extra = [dict(r) for r in out if r["variant"] == "EXTRACT"][:5]
        for r in extra:
            r["correct"] = 1 - r["correct"]
        out += extra
    return pd.DataFrame(out)

class CfgE(Cfg):
    MIN_DISTINCT_VERBAL=3; CALIBRATOR="auto"; ISOTONIC_MIN_N=200
    GROUND_TRUTH_VARIANT="EXTRACT"; COMMIT_CELLS_OUTSIDE_BAND=True
_cfgE = CfgE()
_committedE = {f"{m}__R1": {"model": m, "tier": "R1", "committed": True, "in_band": True,
                            "accuracy": 0.5} for m in _MODELS2}

_gradedE = _graded_rows()
_sigE, _metaE = NS["assemble_signals"](_bankE, _gradedE, pd.DataFrame(), pd.DataFrame(),
                                       _committedE, _cfgE, gate3={})
check("assemble_signals returns rows", len(_sigE) > 0, f"-> {len(_sigE)}")

# B is constructed to have the best Brier of any format. It must still lose.
_fs = _metaE["format_stats"]
check("B has the best Brier in this fixture",
      _fs["B"]["brier"] == min(v["brier"] for v in _fs.values()),
      f"-> {dict((k, round(v['brier'], 4)) for k, v in _fs.items())}; if B does not win, "
      f"the test is not exercising the defect")
check("canonical is NOT B (end to end)", _metaE["canonical_format"] != "B",
      f"-> chose {_metaE['canonical_format']}")
check("canonical is label-free", _metaE["canonical_format"] in NS["LABEL_FREE_FORMATS"],
      f"-> {_metaE['canonical_format']}")
check("B tagged supervised_reference", _fs["B"]["role"] == "supervised_reference",
      f"-> {_fs['B'].get('role')}")
check("B is not eligible", _fs["B"]["eligible"] is False, f"-> {_fs['B']}")
# The exclusion must be because B is label-derived, NOT because it failed the
# distinctness pre-flight — otherwise removing the label-free filter would
# change nothing and this whole block would prove nothing.
check("B clears the distinctness pre-flight",
      _fs["B"]["n_distinct"] >= _cfgE.MIN_DISTINCT_VERBAL and _fs["B"]["n"] >= 20,
      f"-> n_distinct={_fs['B']['n_distinct']}; B is being excluded by §8·6 degeneracy, so "
      f"this fixture cannot detect the label-derivation defect")
check("B is excluded for being label-derived", _fs["B"]["label_free"] is False,
      f"-> {_fs['B']}")
check("ground-truth variant recorded", _metaE.get("canonical_format") is not None)
check("delta column present", "delta_verbal_behavioral" in _sigE.columns)
check("internal_excluded present", "internal_excluded" in _sigE.columns)
# ...and it must mean something. A row with no internal score is excluded; the
# reason distinguishes "Gate 3 removed the cell" from "no probe score", which
# the GLMM's single missingness indicator otherwise lumps together.
_ie = _sigE["internal_excluded"]
check("internal_excluded tracks internal_raw",
      bool((_ie == _sigE["internal_raw"].isna()).all()),
      f"-> {int((_ie != _sigE['internal_raw'].isna()).sum())} rows disagree")
check("internal exclusion carries a reason",
      "internal_excluded_reason" in _sigE.columns
      and set(_sigE.loc[_ie, "internal_excluded_reason"].dropna().unique())
          <= {"gate3_failed", "no_probe_score"},
      f"-> {_sigE.get('internal_excluded_reason', pd.Series(dtype=object)).unique()[:5]}")
# The label-free backstop is defence in depth behind the eligible/fallback
# comprehensions. Assert the raise EXISTS (AST, not a string grep) — a test
# cannot reach it without also breaking the comprehensions it backs up.
_asm = next(x for x in _TREE.body
            if isinstance(x, _ast.FunctionDef) and x.name == "assemble_signals")
_guards = [n for n in _ast.walk(_asm)
           if isinstance(n, _ast.If)
           and "LABEL_FREE_FORMATS" in _ast.unparse(n.test)
           and any(isinstance(b, _ast.Raise) for b in _ast.walk(n))]
check("label-free backstop raise is live",
      bool(_guards),
      "-> no `if canonical not in LABEL_FREE_FORMATS: raise` guard; checking for the raise "
      "alone is not enough, since it survives inside a dead `if False:` branch")

# Ground-truth conflicts must be LOGGED, not silently coin-flipped.
_evts = []
class _Spy:
    def log(self, ev, **kw): _evts.append(ev)
_realLOG2 = NS["LOG"]; NS["LOG"] = _Spy()
try:
    _n_before = len(_sigE)
    _sigC, _ = NS["assemble_signals"](_bankE, _graded_rows(dup_conflict=True), pd.DataFrame(),
                                      pd.DataFrame(), _committedE, _cfgE, gate3={})
finally:
    NS["LOG"] = _realLOG2
check("conflicting ground truth is logged", "ground_truth_conflicts" in _evts,
      f"-> events {sorted(set(_evts))}; a duplicate key graded two ways became an "
      f"unrecorded coin flip")
check("conflicts do not inflate rows", len(_sigC) == _n_before,
      f"-> {len(_sigC)} vs {_n_before}")

# ======================================================================
# WIRING: run_pipeline's call sites.
#
# Every gate above is enforced inside a function, and every one of them can be
# reverted by deleting a keyword argument at the single place it is called.
# `run_pipeline` needs a GPU and a run directory, so no test can execute it —
# which is exactly why a wiring mistake there is invisible. This walks its AST
# instead. Structural, but the structure IS the property: these are assertions
# about the call graph, not greps for a string.
# ======================================================================
_rp = next(x for x in _TREE.body
           if isinstance(x, _ast.FunctionDef) and x.name == "run_pipeline")

def _calls_to(fn_node, name):
    return [c for c in _ast.walk(fn_node) if isinstance(c, _ast.Call)
            and _ast.unparse(c.func) == name]

def _kwargs_of(call):
    return {k.arg for k in call.keywords if k.arg}

for fname, required in [
    ("test_h4_depth", {"gate3"}),
    ("test_h2_quadrants", {"gate1_pass", "gate3"}),
    ("assemble_signals", {"gate3"}),
    ("hierarchical_regression", {"bank"}),
    ("stage_report", {"popularity", "matched"}),
]:
    cs = _calls_to(_rp, fname)
    check(f"run_pipeline calls {fname}", len(cs) == 1, f"-> {len(cs)} call sites")
    if cs:
        missing = required - _kwargs_of(cs[0])
        check(f"run_pipeline passes {sorted(required)} to {fname}", not missing,
              f"-> missing {sorted(missing)}; deleting that keyword reverts the fix silently "
              f"and nothing else would notice")

# fig4 must be drawn from the FILTERED sweep, not the raw one.
_filt = [n for n in _ast.walk(_rp) if isinstance(n, _ast.Assign)
         and isinstance(n.value, _ast.Call)
         and _ast.unparse(n.value.func) == "gate3_filter_sweep"]
check("run_pipeline filters the sweep for figures", len(_filt) == 1,
      f"-> {len(_filt)} gate3_filter_sweep assignments")
if _filt:
    _fig_names = {t.id for t in _ast.walk(_filt[0].targets[0]) if isinstance(t, _ast.Name)}
    _figcalls = _calls_to(_rp, "stage_figures")
    check("stage_figures receives it", len(_figcalls) == 1, f"-> {len(_figcalls)} call sites")
    if _figcalls:
        _args = [_ast.unparse(a) for a in _figcalls[0].args]
        check("fig4 uses the filtered sweep",
              any(a in _fig_names for a in _args),
              f"-> stage_figures{tuple(_args)} does not receive the filtered sweep; the depth "
              f"figure would show cells H4 excluded")

# The judge must run BEFORE the stats block, or H2's Gate 1 conjunct has no
# value to read and silently degrades to None.
_body = [_ast.unparse(n) for n in _rp.body]
_judge_at = next((i for i, t in enumerate(_body) if "stage_judge(" in t), None)
_stats_at = next((i for i, t in enumerate(_body) if "test_h2_quadrants(" in t), None)
check("judge runs before the stats block",
      _judge_at is not None and _stats_at is not None and _judge_at < _stats_at,
      f"-> judge at {_judge_at}, H2 at {_stats_at}")

print("FAILURES:" if fails else "ALL ANALYSIS TESTS PASSED")
for f in fails: print("  -", f)
sys.exit(1 if fails else 0)
