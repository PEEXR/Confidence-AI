#!/usr/bin/env python3
"""Re-run the CPU analysis stages on an existing run's `graded.parquet`.

    python audit_docs/reanalyse.py [--source results] [--out results_repaired]

Answers one question without booking a GPU: **does anything survive the audit
repair?** The generation stages are the expensive half of the pipeline and
their output is already on disk, so the verbal axis, the entropy
normalisation, and the H1/H2/H3 tests can all be recomputed today.

WHAT THIS CAN REBUILD
  * the canonical verbal axis        - graded.parquet carries A's stated
                                       numbers, B's buckets and C's decisions
  * semantic entropy                 - the 10 SAMPLE rows per question are all
                                       present, so clustering, math equivalence,
                                       and the fixed log(N_SAMPLES) denominator apply
  * calibration, H0, H1, H2, H3      - all downstream of the two above
  * H2 question-feature tests        - question text recovered from HuggingFace
                                       datasets so within-cell tests and feature
                                       replication can execute
  * difficulty matching              - cross-model correctness bins within tiers
  * multi-level GLMM                 - hierarchical logistic regression with
                                       model, tier, model x tier, and question
                                       random intercepts
  * popularity gradient contrast     - PopQA R1 vs R2 analysis

WHAT IT CANNOT, AND WHY
  * log-probability weighting        - needs the raw generations, which are not
                                       in the repo. Entropy is COUNT-weighted
                                       here, exactly as `cluster_mass` falls
                                       back for legacy checkpoints.
  * the internal probe, H4, Gate 3   - needs the activation shards, also absent
                                       (and they were captured by the broken
                                       tap anyway). `internal_cal` is NaN
                                       throughout, which is the honest value.

Nothing in the source tree is modified; everything lands in `--out`.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib.util
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import scipy.stats as sps

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Load the pipeline's pure functions without importing the module.
#
# `confidence_pipeline.py` installs dependencies, builds output paths and runs
# the whole pipeline at import time. Parsing it and exec'ing the definitions we
# need keeps this script honest: it exercises the SAME code the notebook runs,
# with I/O stubbed, rather than a reimplementation that could drift from it.
# ---------------------------------------------------------------------------
def load_pipeline(out_dir: Path) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    events: list[tuple[str, dict]] = []
    failed: list[tuple[str, str]] = []

    class _Paths(dict):
        def __getitem__(self, k):
            p = out_dir / ("derived" if k in ("derived", "acts") else k)
            p.mkdir(parents=True, exist_ok=True)
            return p

    class _Log:
        def log(self, ev, **kw):
            events.append((ev, kw))

    def json_write(path, obj):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=1, default=str))

    def json_read(path, default=None):
        p = Path(path)
        return json.loads(p.read_text()) if p.exists() else default

    ns: dict[str, Any] = dict(
        np=np, pd=pd, sps=sps, re=re, json=json, hashlib=__import__("hashlib"),
        warnings=warnings, Counter=Counter, defaultdict=defaultdict, Path=Path,
        Any=Any, Callable=Callable, Sequence=Sequence,
        dataclass=dataclasses.dataclass, field=dataclasses.field,
        asdict=dataclasses.asdict, replace=dataclasses.replace,
        # Call-time-only globals the dataclass methods close over.
        PLATFORM="local", DEVICES={"cuda": False, "bf16": False, "max_vram_gb": 0},
        torch=None,
        PATHS=_Paths(), LOG=_Log(), json_write=json_write, json_read=json_read,
        LogisticRegression=LogisticRegression, roc_auc_score=roc_auc_score,
        make_pipeline=make_pipeline, StandardScaler=StandardScaler,
        IsotonicRegression=IsotonicRegression, TfidfVectorizer=TfidfVectorizer,
        HAS_MATH_VERIFY=importlib.util.find_spec("math_verify") is not None,
        tqdm=lambda x, **k: x,
        _events=events,
    )

    src = (ROOT / "confidence_pipeline.py").read_text()
    tree = ast.parse(src)
    SKIP = {"NLIGrader", "ActivationTap", "LoadedModel", "Checkpoint", "RunLog",
            "JudgeConfig", "SeqLogProbAccumulator"}
    WANT_ASSIGN = {
        "BUCKETS", "BUCKET_FIXED_VALUES", "LABEL_FREE_FORMATS", "VERBAL_FORMATS",
        "SIGNALS", "QUADRANT_TAXONOMY", "QUADRANT_LABELS", "TIER_SPECS",
        "MODEL_SPECS", "MATH_SUBS", "ARTICLES", "PUNCT_RE", "ANSWER_STYLE",
    }
    for node in tree.body:
        try:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
                if {getattr(t, "id", "") for t in tgts} & WANT_ASSIGN:
                    exec(compile(ast.Module([node], []), "<p>", "exec"), ns)
            elif isinstance(node, ast.ClassDef) and node.name not in SKIP:
                exec(compile(ast.Module([node], []), "<p>", "exec"), ns)
            elif isinstance(node, ast.FunctionDef):
                exec(compile(ast.Module([node], []), "<p>", "exec"), ns)
        except Exception as exc:                          # noqa: BLE001
            failed.append((getattr(node, "name", "?"), f"{type(exc).__name__}: {exc}"))
    m = re.search(r"^BET_GAIN,\s*BET_LOSS\s*=\s*(.+)$", src, re.M)
    if m:
        ns["BET_GAIN"], ns["BET_LOSS"] = eval(m.group(1))

    # Anything this script actually calls must have loaded. A silent miss here
    # would surface much later as a confusing KeyError.
    REQUIRED = ["Config", "assemble_signals", "cluster_answers", "cluster_mass",
                "test_h0_format_agreement", "test_h1_signal_calibration",
                "test_h2_quadrants", "test_h3_base_vs_instruct", "abstention_split",
                "cell_id", "empirical_bucket_map", "verbal_scores", "fit_calibrator",
                "ece", "brier", "murphy_decomposition", "bootstrap_diff_ci",
                "hierarchical_regression", "difficulty_matched_view", "popularity_contrast"]
    missing = [n for n in REQUIRED if n not in ns]
    if missing:
        detail = {n: e for n, e in failed if n in missing}
        raise RuntimeError(f"could not load from confidence_pipeline.py: {missing}\n{detail}")
    return ns


# ---------------------------------------------------------------------------
def recover_questions(tiers: Sequence[str]) -> dict[str, str]:
    """Recover question text from HuggingFace datasets by QID.

    The question bank was randomly sampled with fixed seeds from public datasets.
    Restoring question text allows question features (is_long, has_number, has_year,
    has_multi_entity) to be evaluated within cells rather than skipped.
    """
    qmap: dict[str, str] = {}
    try:
        from datasets import load_dataset
    except ImportError:
        print("notice: `datasets` package not available; skipping question recovery", file=sys.stderr)
        return qmap

    for tier in tiers:
        try:
            if tier in ("R1", "R2") and not any(k.startswith("R1-") for k in qmap):
                ds = load_dataset("akariasai/PopQA", split="test")
                for i, r in enumerate(ds):
                    q = str(r.get("question", "")).strip()
                    qmap[f"R1-{r.get('id', i)}"] = q
                    qmap[f"R2-{r.get('id', i)}"] = q
            elif tier == "R3" and not any(k.startswith("R3-") for k in qmap):
                ds = load_dataset("basicv8vc/SimpleQA", split="test")
                qkey = "problem" if "problem" in ds.column_names else "question"
                for i, r in enumerate(ds):
                    qmap[f"R3-{i}"] = str(r.get(qkey, "")).strip()
            elif tier == "C1" and not any(k.startswith("C1-") for k in qmap):
                ds = load_dataset("openai/gsm8k", "main", split="test")
                for i, r in enumerate(ds):
                    qmap[f"C1-{i}"] = str(r.get("question", "")).strip()
            elif tier in ("C2", "C3") and not any(k.startswith("C2-") for k in qmap):
                ds = load_dataset("qwedsacf/competition_math", split="train")
                for i, r in enumerate(ds):
                    p = str(r.get("problem", "")).strip()
                    qmap[f"C2-{i}"] = p
                    qmap[f"C3-{i}"] = p
        except Exception as exc:                          # noqa: BLE001
            print(f"warning: could not recover question text for tier {tier}: {exc}", file=sys.stderr)
    return qmap


# ---------------------------------------------------------------------------
def entropy_from_graded(graded: pd.DataFrame, ns: dict, cfg) -> pd.DataFrame:
    """Semantic entropy rebuilt from graded SAMPLE rows.

    `stage_entropy` reads the raw sample JSONL, which is not in the repo. The
    parsed answers survive in `graded.parquet`, so the clustering and the
    normalisation are the pipeline's own; only the input source differs.
    Count-weighted throughout: no log-probabilities were recorded by the run
    that produced this data, which is exactly the legacy path `cluster_mass`
    already falls back to.
    """
    cluster_answers, cluster_mass = ns["cluster_answers"], ns["cluster_mass"]
    n_req = int(cfg.N_SAMPLES)
    H_max = float(np.log(n_req))
    rows = []
    s = graded[graded.variant == "SAMPLE"]
    for (model, tier, qid), g in s.groupby(["model", "tier", "qid"], sort=False):
        answers = [a for a in g["answer"].tolist() if isinstance(a, str) and a.strip()]
        form = g["answer_form"].iloc[0]
        split = g["split"].iloc[0]
        n = len(answers)
        low = n < int(cfg.ENTROPY_MIN_VALID)
        if n == 0:
            rows.append(dict(model=model, tier=tier, qid=qid, split=split, n_valid=0,
                             n_requested=n_req, n_clusters=np.nan, entropy=np.nan,
                             entropy_max=H_max, confidence_behavioral=np.nan,
                             modal_share=np.nan, modal_mass=np.nan,
                             low_valid=True, lp_weighted=False))
            continue
        labels = cluster_answers(answers, form, cfg, None)
        sizes, lp_w = cluster_mass(labels, None, cfg)
        p = sizes / sizes.sum()
        H = float(sps.entropy(p))
        rows.append(dict(
            model=model, tier=tier, qid=qid, split=split, n_valid=n, n_requested=n_req,
            n_clusters=(np.nan if low else int(len(sizes))),
            entropy=(np.nan if low else H), entropy_max=H_max,
            confidence_behavioral=(np.nan if low else float(1 - H / H_max)),
            modal_share=(np.nan if low else float(sizes.max() / sizes.sum())),
            modal_mass=(np.nan if low else float(sizes.max() / sizes.sum())),
            low_valid=bool(low), lp_weighted=bool(lp_w)))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results")
    ap.add_argument("--out", default="results_repaired")
    ap.add_argument("--skip-question-recovery", action="store_true",
                    help="Skip downloading question texts from HuggingFace")
    a = ap.parse_args()
    src_dir, out_dir = ROOT / a.source, ROOT / a.out
    graded_path = src_dir / "derived" / "graded.parquet"
    if not graded_path.exists():
        print(f"no graded.parquet under {src_dir}", file=sys.stderr)
        return 1
    (out_dir / "derived").mkdir(parents=True, exist_ok=True)

    ns = load_pipeline(out_dir)
    Config = ns["Config"]
    cfg = ns["replace"](Config(), RUN_NAME="reanalysis", N_BOOTSTRAP=2000)

    graded = pd.read_parquet(graded_path)
    print(f"graded rows          : {len(graded):,}  ({graded.groupby(['model','tier']).ngroups} cells)")

    tiers = sorted(graded["tier"].unique())
    qmap: dict[str, str] = {}
    if not a.skip_question_recovery:
        print("recovering questions : querying HuggingFace datasets...")
        qmap = recover_questions(tiers)
        n_matched = sum(1 for q in graded["qid"].unique() if q in qmap)
        print(f"questions recovered  : {n_matched:,} / {graded['qid'].nunique():,} unique QIDs")

    bank: dict[str, list[dict]] = {}
    for tier, g in graded.groupby("tier"):
        seen = g.drop_duplicates("qid")
        bank[tier] = [dict(qid=r.qid, split=r.split, question=qmap.get(r.qid, ""),
                           family=r.family, answer_form=r.answer_form)
                      for r in seen.itertuples()]
    has_qtext = any(r["question"] for v in bank.values() for r in v)
    print(f"bank                 : {sum(len(v) for v in bank.values()):,} questions, "
          f"{len(bank)} tiers  ({'with question text -> within-cell H2 features enabled' if has_qtext else 'no question text -> H2 features skipped'})")

    cached_entropy = out_dir / "derived" / "entropy.parquet"
    if cached_entropy.exists():
        entropy = pd.read_parquet(cached_entropy)
        print(f"entropy loaded (cache): {len(entropy):,} questions")
    else:
        entropy = entropy_from_graded(graded, ns, cfg)
        n_nan = int(entropy["confidence_behavioral"].isna().sum())
        print(f"entropy rebuilt      : {len(entropy):,} questions, "
              f"{n_nan} NaN under ENTROPY_MIN_VALID={cfg.ENTROPY_MIN_VALID} "
              f"({n_nan/max(len(entropy),1):.1%})")
        entropy.to_parquet(cached_entropy, index=False)

    committed = {ns["cell_id"](m, t): dict(model=m, tier=t, committed=True,
                                           in_band=True, accuracy=float(g["correct"].mean()))
                 for (m, t), g in graded.groupby(["model", "tier"])}

    # Empty sweep: the probe cannot be rebuilt, so the internal signal is NaN.
    signals, meta = ns["assemble_signals"](bank, graded, entropy, pd.DataFrame(),
                                           committed, cfg, gate3={})
    print(f"signals              : {len(signals):,} rows")
    print(f"canonical format     : {meta['canonical_format']}  "
          f"(was: {json.loads((src_dir/'derived'/'calibration_meta.json').read_text()).get('canonical_format')})")

    h0 = ns["test_h0_format_agreement"](meta["verbal_long"], cfg)
    h1 = ns["test_h1_signal_calibration"](signals, cfg)
    manual_sheet = src_dir / "tables" / "gate1_manual_check_sheet.csv"
    gate1 = ns["ingest_manual_sheet"](manual_sheet, cfg) if manual_sheet.exists() else {}
    gate1_pass = gate1.get("passes")
    if gate1.get("available"):
        print(f"Gate 1 manual check  : {gate1['n_labelled']}/{gate1['n_rows']} rows, agreement={gate1['agreement']:.1%}, passes={gate1_pass}")
    h2 = ns["test_h2_quadrants"](signals, bank, cfg, gate1_pass=gate1_pass, gate3={})
    abst = ns["abstention_split"](graded, cfg)
    h3 = ns["test_h3_base_vs_instruct"](signals, abst, cfg)

    print("running GLMM         : hierarchical_regression...")
    hlr = ns["hierarchical_regression"](signals, cfg, bank=bank)
    print(f"GLMM completed       : method={hlr.get('method')} available={hlr.get('available')}")

    print("running matching     : difficulty_matched_view...")
    matched = ns["difficulty_matched_view"](signals, cfg)
    print(f"matching completed   : tiers={len(matched.get('per_tier', {}))} available={matched.get('available')}")

    print("running popularity   : popularity_contrast...")
    pop = ns["popularity_contrast"](signals, graded, cfg)
    print(f"popularity completed : available={pop.get('available')}")

    signals.to_parquet(out_dir / "derived" / "signals.parquet", index=False)
    json_dump = lambda name, obj: (out_dir / "derived" / name).write_text(
        json.dumps(obj, indent=1, default=str))
    json_dump("h0_gate2.json", h0)
    json_dump("h1_calibration.json", h1)
    json_dump("h2_quadrants.json", h2)
    json_dump("h3_model_delta.json", h3)
    json_dump("hierarchical_regression.json", hlr)
    json_dump("difficulty_matched.json", matched)
    json_dump("popularity_gradient.json", pop)
    json_dump("calibration_meta.json", {k: v for k, v in meta.items() if k != "verbal_long"})
    json_dump("_provenance.json", {
        "source": str(src_dir), "generated_by": "audit_docs/reanalyse.py",
        "internal_signal": "UNAVAILABLE - activation shards absent; probe/H4/Gate 3 not rebuilt",
        "entropy_weighting": "counts - no sequence log-probs in this run's artefacts",
        "h2_features": ("AVAILABLE - question text recovered from HF datasets" if has_qtext
                        else "UNAVAILABLE - no question text survives"),
        "hierarchical_regression": f"COMPLETED - method={hlr.get('method')}",
        "difficulty_matched": f"COMPLETED - tiers={len(matched.get('per_tier', {}))}",
        "popularity_contrast": f"available={pop.get('available')}",
    })
    print(f"\nwritten to {out_dir.relative_to(ROOT)}/derived/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
