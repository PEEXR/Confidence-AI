"""Load the pipeline's pure-analysis functions with I/O stubbed out."""
import ast, sys, types, re, hashlib, warnings, json as _json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd, scipy.stats as sps
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "confidence_pipeline.py").read_text()
tree = ast.parse(SRC)

WRITES = {}
class _P(dict):
    def __getitem__(self, k):
        p = Path(str(Path(__file__).resolve().parent / "_out")) / k
        p.mkdir(parents=True, exist_ok=True); return p
class _LOG:
    def __init__(s): s.events=[]
    def log(s, ev, **kw): s.events.append((ev, kw))
LOG = _LOG()
def json_write(path, obj):
    WRITES[Path(path).name] = obj
    Path(path).write_text(_json.dumps(obj, indent=1, default=str))
def json_read(path, default=None):
    p = Path(path)
    return _json.loads(p.read_text()) if p.exists() else default

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.feature_extraction.text import TfidfVectorizer

NS = dict(np=np, pd=pd, sps=sps, re=re, json=_json, hashlib=hashlib, warnings=warnings,
          replace=__import__("dataclasses").replace,
          Counter=Counter, defaultdict=defaultdict,
          Path=Path, Any=Any, Callable=Callable, Sequence=Sequence,
          PATHS=_P(), LOG=LOG, json_write=json_write, json_read=json_read,
          LogisticRegression=LogisticRegression, roc_auc_score=roc_auc_score,
          make_pipeline=make_pipeline, StandardScaler=StandardScaler,
          IsotonicRegression=IsotonicRegression, TfidfVectorizer=TfidfVectorizer,
          HAS_MATH_VERIFY=False, tqdm=lambda x, **k: x)

SKIP_CLASSES = {"NLIGrader", "ActivationTap", "LoadedModel", "Checkpoint", "RunLog", "JudgeConfig"}
NEED_ASSIGN = {"BUCKETS", "BUCKET_FIXED_VALUES", "LABEL_FREE_FORMATS", "VERBAL_FORMATS",
               "SIGNALS", "QUADRANT_TAXONOMY", "QUADRANT_LABELS", "TIER_SPECS", "MODEL_SPECS",
               "BET_GAIN", "BET_LOSS", "MATH_SUBS", "ARTICLES", "PUNCT_RE"}
for node in tree.body:
    try:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {getattr(t, "id", "") for t in tgts}
            if names & NEED_ASSIGN:
                exec(compile(ast.Module([node], []), "<x>", "exec"), NS)
        elif isinstance(node, ast.ClassDef) and node.name not in SKIP_CLASSES:
            exec(compile(ast.Module([node], []), "<x>", "exec"), NS)
        elif isinstance(node, ast.FunctionDef):
            exec(compile(ast.Module([node], []), "<x>", "exec"), NS)
    except Exception:
        pass
# BET_GAIN/BET_LOSS are a tuple-assign; catch it explicitly
m = re.search(r"^BET_GAIN,\s*BET_LOSS\s*=\s*(.+)$", SRC, re.M)
if m: NS["BET_GAIN"], NS["BET_LOSS"] = eval(m.group(1))
