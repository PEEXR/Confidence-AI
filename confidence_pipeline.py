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
try:
    from sympy.parsing.sympy_parser import parse_expr as sym_parse_expr
    SYMPY_OK = True
except Exception:                                          # pragma: no cover
    sym_parse_expr = None
    SYMPY_OK = False


def detect_platform() -> str:
    """kaggle | molab | colab | local — from filesystem + env markers."""
    if Path("/kaggle/working").exists():
        return "kaggle"
    if os.environ.get("MARIMO_MOLAB") or Path("/molab").exists():
        return "molab"
    if "MARIMO_ROOT" in os.environ or importlib.util.find_spec("marimo") is not None:
        # marimo is present; molab is the hosted flavour. Treat as molab only
        # when there is a GPU big enough to be the Blackwell box.
        if torch.cuda.is_available():
            try:
                if torch.cuda.get_device_properties(0).total_memory > 60e9:
                    return "molab"
            except Exception:
                pass
    if importlib.util.find_spec("google.colab") is not None:
        return "colab"
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
# CELL 1b — session environment: secrets + run identity (ENV ONLY)
#
# Set these HERE so they exist before any Hub download or provenance stamp.
# Values stay on the molab/kaggle machine — NEVER commit them to the repo.
# ============================================================================

# HuggingFace token (Read scope is enough): faster downloads and access to
# gated repos like the Qwen family. Paste between the quotes; leave empty to
# skip (public repos still work, rate-limited).
_HF_TOKEN = ""
if _HF_TOKEN:
    os.environ["HF_TOKEN"] = _HF_TOKEN
del _HF_TOKEN

# Binds every artifact to a version label when no .git exists on the platform
# (audit fix for code_sha="nogit"). Change per release, e.g. a git sha.
os.environ.setdefault("CODE_SHA", "v3-qwen35-audit-fixes")

# Optimized linear-attention kernels for the Qwen 3.5 hybrid layers
# (flash-linear-attention + causal-conv1d). Speed-only: identical results
# either way. Installed BEST-EFFORT and SEPARATELY — flash-linear-attention
# is pure Python/Triton and effectively always succeeds, while causal-conv1d
# usually has no prebuilt wheel for the session's torch/CUDA/Python combo.
#
# For causal-conv1d the fallback chain is:
#   1. plain pip install (hits GitHub prebuilt wheels when one exists)
#   2. source build against a PIP-PROVIDED CUDA toolchain: Molab images ship
#      no nvcc, and upstream setup.py then dies on an unbound
#      `bare_metal_version`. NVIDIA publishes nvcc + headers on PyPI
#      (nvidia-cuda-nvcc / -runtime / -cccl), so we assemble CUDA_HOME from
#      them and retry with --no-build-isolation.
# Set INSTALL_LINEAR_ATTN_KERNELS=0 to skip entirely.

def _kernel_importable(mod_name):
    """True only if the module IMPORTS cleanly — a half-built extension that
    is findable but crashes on import must not count as installed."""
    try:
        importlib.import_module(mod_name)
        return True
    except Exception:                                      # noqa: BLE001
        return False


def _pip_show_location(pkg_name):
    """Exact site-packages dir pip installed <pkg_name> into, via
    `pip show`. Needed because `import nvidia` is a NAMESPACE package whose
    first __path__ entry may be torch's bundled nvidia/ libs — probing only
    that masked the freshly-installed toolchain (v3 bug)."""
    try:
        _r = subprocess.run([sys.executable, "-m", "pip", "show", pkg_name],
                            capture_output=True, text=True, timeout=120)
        if _r.returncode != 0:
            return None
        import re as _re
        _m = _re.search(r"^Location:\s*(.+)$", _r.stdout, _re.M)
        return Path(_m.group(1).strip()) if _m else None
    except Exception:                                      # noqa: BLE001
        return None


if os.environ.get("INSTALL_LINEAR_ATTN_KERNELS", "1") != "0":
    if not _kernel_importable("fla"):
        print("[cell1b] installing flash-linear-attention ...")
        try:
            _r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-input",
                 "flash-linear-attention"],
                capture_output=True, text=True, timeout=600)
            print("[cell1b] flash-linear-attention: "
                  + ("installed" if _r.returncode == 0 else
                     "FAILED:\n" + (_r.stderr or _r.stdout or "")[-400:]))
        except Exception as _e:                            # noqa: BLE001
            print(f"[cell1b] flash-linear-attention install failed ({_e})")

    if not _kernel_importable("causal_conv1d"):
        # Attempt 0 — reuse a wheel built by an EARLIER session (cached next
        # to the notebook): seconds instead of another toolchain compile.
        _cc1d_env = os.environ.copy()
        for _whl in sorted(Path.cwd().glob("causal_conv1d-*.whl"), reverse=True):
            print(f"[cell1b] trying cached wheel {_whl.name} ...")
            try:
                _r = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "--no-input",
                     "--force-reinstall", "--no-deps", str(_whl)],
                    capture_output=True, text=True, timeout=600, env=_cc1d_env)
            except Exception as _e:                        # noqa: BLE001
                print(f"[cell1b] cached-wheel install failed ({_e})")
                continue
            if _r.returncode == 0 and _kernel_importable("causal_conv1d"):
                print("[cell1b] causal-conv1d: restored from cached wheel")
                break
            print(f"[cell1b] cached wheel unusable ({_whl.name})")

    if not _kernel_importable("causal_conv1d"):
        print("[cell1b] installing causal-conv1d (prebuilt wheel or source) ...")
        _cc1d_env = os.environ.copy()
        try:
            _r = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-input",
                 "causal-conv1d"],
                capture_output=True, text=True, timeout=900, env=_cc1d_env)
            _ok = _r.returncode == 0
            print("[cell1b] causal-conv1d (wheel): "
                  + ("installed" if _ok else
                     "no matching prebuilt wheel\n" + (_r.stderr or "")[-200:]))
        except Exception as _e:                            # noqa: BLE001
            _ok = False
            print(f"[cell1b] causal-conv1d wheel attempt failed ({_e})")

        if not _ok:
            # ---- causal-conv1d source build, best-effort -----------------
            # Attempt order (every failure is non-fatal -> slow path):
            #   a) SYSTEM toolkit (/usr/local/cuda*, /opt/cuda*) — exact match,
            #      zero download;
            #   b) pip toolchain PINNED to torch.version.cuda (nvcc-latest vs
            #      torch-cu13.0 header skew caused a real compile failure);
            #   c) pip toolchain unpinned.
            # NVIDIA wheels >=13.x unpack to one versioned root
            # nvidia/cu13/{bin/nvcc,include/,lib/}; older ones used
            # nvidia/cuda_nvcc. `import nvidia` is a namespace package that may
            # lead with torch's bundled libs, so roots come from `pip show`
            # plus every site-packages dir — never __path__[0] alone.
            def _nvidia_roots():
                """Every candidate nvidia/ root: pip-show locations for the
                toolchain packages, all site-packages dirs, and the namespace
                package paths. `import nvidia` alone is NOT enough — its first
                __path__ entry can be torch's bundled lib-only tree."""
                _roots = []
                for _pkg in ("nvidia-cuda-nvcc", "nvidia-cuda-runtime"):
                    _loc = _pip_show_location(_pkg)
                    if _loc is not None:
                        _roots.append(Path(_loc) / "nvidia")
                try:
                    import site as _site_mod
                    for _sp in list(_site_mod.getsitepackages()) + \
                            [_site_mod.getusersitepackages()]:
                        _cand = Path(_sp) / "nvidia"
                        if _cand.is_dir():
                            _roots.append(_cand)
                    for _p in Path(importlib.import_module("nvidia").__path__):
                        _roots.append(Path(_p))
                except Exception:                          # noqa: BLE001
                    pass
                _seen, _out = set(), []
                for _r in _roots:
                    if str(_r) in _seen:
                        continue
                    _seen.add(str(_r))
                    if _r.is_dir():
                        _out.append(_r)
                return _out

            def _find_nvcc():
                for _rt in _nvidia_roots():
                    for _cand in sorted(_rt.glob("cu1*")) + [_rt / "cuda_nvcc", _rt]:
                        if (_cand / "bin" / "nvcc").is_file():
                            return _cand
                return None

            def _nvcc_banner(home):
                _v = subprocess.run([str(home / "bin" / "nvcc"), "-V"],
                                    capture_output=True, text=True, timeout=120)
                if _v.returncode != 0:
                    print("[cell1b] nvcc probe failed:\n" + (_v.stderr or "")[-300:])
                    return False
                print("[cell1b] " + "\n[cell1b] ".join(
                    ln for ln in _v.stdout.strip().splitlines() if ln.strip()))
                return True

            def _prep_home(home):
                """Make CUDA_HOME self-sufficient: exec bits on bin/, and merge
                headers + libcudart from EVERY nvidia root — the pinned nvcc
                root can differ from where torch's matching 13.0.x runtime
                headers live, and missing/mismatched headers were a prime
                compile-failure suspect. Also provides lib64/ for the linker."""
                for _b in (home / "bin").glob("*"):
                    try:
                        _b.chmod(_b.stat().st_mode | 0o111)
                    except OSError:
                        pass
                _inc = home / "include"
                _lib64 = home / "lib64"
                _inc.mkdir(parents=True, exist_ok=True)
                _lib64.mkdir(parents=True, exist_ok=True)
                for _rt in _nvidia_roots():
                    for _base in [_rt] + sorted(_rt.glob("cu1*")) + [_rt / "cuda_nvcc"]:
                        _src_inc = _base / "include"
                        if _src_inc.is_dir() and _src_inc.resolve() != _inc.resolve():
                            for _f in _src_inc.rglob("*"):
                                if _f.is_file():
                                    _dst = _inc / _f.relative_to(_src_inc)
                                    _dst.parent.mkdir(parents=True, exist_ok=True)
                                    shutil.copy2(_f, _dst)
                        for _ld in ("lib", "lib64"):
                            _d = _base / _ld
                            if _d.is_dir():
                                for _so in _d.glob("libcudart*.so*"):
                                    shutil.copy2(_so, _lib64 / _so.name)

            def _attempt_build(home):
                """Build via setup.py bdist_wheel DIRECTLY: pip -q suppresses
                ninja/nvcc output and torch's failure wrapper cannot surface
                it ("no good way to get it here") — which is exactly why the
                earlier failures showed no compiler diagnostics. Running the
                build ourselves streams every nvcc line into our capture, and
                the produced wheel is cached beside the notebook for instant
                reuse in later sessions."""
                if not _nvcc_banner(home):
                    return False
                _env = os.environ.copy()
                _env["CUDA_HOME"] = str(home)
                _env["PATH"] = str(home / "bin") + os.pathsep + _env.get("PATH", "")
                _env.setdefault("TORCH_CUDA_ARCH_LIST",
                                "7.5;8.0;8.6;9.0;10.0+PTX;12.0")
                _env["MAX_JOBS"] = "2"     # RAM-hungry nvcc jobs; OOM-kill
                                           # masquerades as 'compile error'
                import tempfile
                _work = Path(tempfile.mkdtemp(prefix="cc1d_build_"))
                _streams = {"out": [], "err": []}

                def _run(cmd, timeout, cwd=None):
                    _r = subprocess.run(cmd, capture_output=True, text=True,
                                        timeout=timeout, cwd=cwd, env=_env)
                    _streams["out"].append(_r.stdout or "")
                    _streams["err"].append(_r.stderr or "")
                    return _r

                def _flush_log():
                    Path("causal_conv1d_build.log").write_text(
                        "=== STDOUT ===\n" + "\n".join(_streams["out"])
                        + "\n\n=== STDERR ===\n" + "\n".join(_streams["err"]),
                        encoding="utf-8")

                def _diagnose():
                    _lines = ("\n".join(_streams["out"])
                              + "\n--- stderr ---\n"
                              + "\n".join(_streams["err"])).splitlines()
                    _hits = []
                    for _i, _l in enumerate(_lines):
                        if re.search(r"\b(error|FAILED|fatal)\b", _l, re.I):
                            _hits.extend(_lines[_i:_i + 4])
                            _hits.append("")
                    print("[cell1b] --- detected diagnostics ---\n"
                          + ("\n".join(_hits)[-3000:]
                             or "(no error lines matched; see full log)"))

                try:
                    _r = _run([sys.executable, "-m", "pip", "download",
                               "causal-conv1d", "--no-deps",
                               "--no-binary", "causal-conv1d",
                               "-d", str(_work)], 600)
                    if _r.returncode != 0:
                        print("[cell1b] sdist download failed:\n"
                              + (_r.stderr or "")[-400:])
                        return False
                    _sdist = next(_work.glob("causal_conv1d-*.tar.gz"))
                    shutil.unpack_archive(_sdist, _work)
                    _src = next(d for d in _work.iterdir() if d.is_dir())
                    print("[cell1b] compiling via setup.py bdist_wheel "
                          "(compiler output fully captured) ...")
                    _r = _run([sys.executable, "setup.py", "bdist_wheel"],
                              2700, cwd=_src)
                    if _r.returncode != 0:
                        _flush_log()
                        print("[cell1b] BUILD FAILED — "
                              "full log -> causal_conv1d_build.log")
                        _diagnose()
                        return False
                    _whls = list((_src / "dist").glob("*.whl"))
                    if not _whls:
                        print("[cell1b] build finished but no wheel produced")
                        return False
                    _cache = Path.cwd() / _whls[0].name
                    shutil.copy2(_whls[0], _cache)
                    print(f"[cell1b] wheel cached for future sessions: {_cache.name}")
                    _r = _run([sys.executable, "-m", "pip", "install", "-q",
                               "--no-input", "--force-reinstall", "--no-deps",
                               str(_cache)], 600)
                    if _r.returncode != 0:
                        print("[cell1b] wheel install failed:\n"
                              + (_r.stderr or "")[-400:])
                        return False
                    if not _kernel_importable("causal_conv1d"):
                        print("[cell1b] built wheel does not import cleanly")
                        return False
                    print("[cell1b] causal-conv1d: built, cached, imported OK")
                    return True
                except Exception as _e:                    # noqa: BLE001
                    _flush_log()
                    print(f"[cell1b] direct build failed ({_e})")
                    _diagnose()
                    return False

            _built = False
            _sys_dirs = [Path("/usr/local/cuda"), Path("/opt/cuda")] + \
                sorted(Path("/usr/local").glob("cuda-*")) + \
                sorted(Path("/usr/local").glob("cuda")) + \
                sorted(Path("/opt").glob("cuda-*"))
            for _d in _sys_dirs:
                if (_d / "bin" / "nvcc").is_file() and _nvcc_banner(_d):
                    print(f"[cell1b] using SYSTEM toolkit at {_d}")
                    _prep_home(_d)
                    _built = _attempt_build(_d)
                    break

            if not _built:
                _tcv = None
                try:
                    _tcv = str(torch.version.cuda)         # e.g. '13.0'
                except Exception:                          # noqa: BLE001
                    pass
                _tool_sets = []
                if _tcv:
                    _tool_sets.append([f"nvidia-cuda-nvcc=={_tcv}.*",
                                       "ninja", "wheel"])
                _tool_sets.append(["nvidia-cuda-nvcc", "nvidia-cuda-runtime",
                                   "nvidia-cuda-cccl",
                                   "nvidia-cuda-profiler-api", "ninja", "wheel"])
                for _pkgs in _tool_sets:
                    _tag = ("version-pinned to torch cu" + _tcv
                            if any("==" in p for p in _pkgs[:1]) else "unpinned latest")
                    print(f"[cell1b] installing pip CUDA toolchain ({_tag}) ...")
                    try:
                        _r = subprocess.run(
                            [sys.executable, "-m", "pip", "install", "-q",
                             "--no-input", *_pkgs],
                            capture_output=True, text=True, timeout=900)
                        if _r.returncode != 0:
                            print("[cell1b] toolchain install failed:\n"
                                  + (_r.stderr or _r.stdout or "")[-300:])
                            continue
                    except Exception as _e:                # noqa: BLE001
                        print(f"[cell1b] toolchain install failed ({_e})")
                        continue
                    _home = _find_nvcc()
                    if _home is None:
                        print("[cell1b] toolchain installed but nvcc not located")
                        continue
                    _prep_home(_home)
                    if not _nvcc_banner(_home):
                        continue
                    _built = _attempt_build(_home)
                    if _built:
                        break
            if not _built:
                print("[cell1b] continuing on the slower torch path; "
                      "results are unaffected")

# Recorded AFTER all attempts: both | fla_only | none.
_state = ("both" if _kernel_importable("fla") and _kernel_importable("causal_conv1d")
          else "fla_only" if _kernel_importable("fla") else "none")
os.environ["LINEAR_ATTN_KERNELS"] = _state
print(f"[cell1b] linear_attn_kernels={_state}")


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
# PLAN §9. `layers` is the TEXT decoder block count; percentile→layer index
# uses it directly (index 0 = embedding output, index L = final block).
#
# Qwen 3.5 family ladder — replaces the Qwen 2.5 grid (audit resolution:
# stronger models raise accuracy on the hardest tiers R3/C2/C3, so more of
# each dataset survives parsing/accuracy floors and contributes rows).
#
# VERIFIED against each checkpoint's config.json on the HF Hub (plain name =
# instruct/chat model, "-Base" suffix = base model; there are no "-Instruct"
# repos in this family):
#   Qwen3.5-0.8B      layers=24 hidden=1024
#   Qwen3.5-2B        layers=24 hidden=2048
#   Qwen3.5-4B        layers=32 hidden=2560
#   Qwen3.5-9B(-Base) layers=32 hidden=4096
# All five are Qwen3_5ForConditionalGeneration (multimodal wrapper, hybrid
# linear/full attention every 4th layer) — the pipeline resolves the text
# stack via text_stack() and re-verifies these numbers at runtime
# (verify_model_specs), so a config drift can never silently mis-tap layers.
MODEL_SPECS: dict[str, dict] = {
    "qwen3.5-0.8b-instruct": dict(
        hf_id="Qwen/Qwen3.5-0.8B", params_b=0.8, layers=24, hidden=1024,
        chat=True, rung="ladder",
    ),
    "qwen3.5-2b-instruct": dict(
        hf_id="Qwen/Qwen3.5-2B", params_b=2.0, layers=24, hidden=2048,
        chat=True, rung="ladder",
    ),
    "qwen3.5-4b-instruct": dict(
        hf_id="Qwen/Qwen3.5-4B", params_b=4.0, layers=32, hidden=2560,
        chat=True, rung="ladder",
    ),
    "qwen3.5-9b-instruct": dict(
        hf_id="Qwen/Qwen3.5-9B", params_b=9.0, layers=32, hidden=4096,
        chat=True, rung="ladder",
    ),
    "qwen3.5-9b-base": dict(
        hf_id="Qwen/Qwen3.5-9B-Base", params_b=9.0, layers=32, hidden=4096,
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
    RUN_NAME: str = "v35_full1000"      # v3 audit fixes + Qwen 3.5 ladder
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
    N_PER_CELL: int = 1000              # committed-cell size (PLAN §3 target 2000)
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
    SAMPLE_SEQ_LOGPROBS: bool = True    # record per-sample sequence log-probs
                                        # (audit fix 3a: Rao-Blackwellized
                                        # semantic-entropy masses)
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
    ACCURACY_BAND: tuple[float, float] = (0.25, 0.80)   # PLAN §3
    COMMIT_CELLS_OUTSIDE_BAND: bool = False             # True = ignore the gate (records the violation)

    # ------------------------------------------------------- probe (§6) ---
    PERCENTILES: tuple[int, ...] = (0, 25, 50, 75, 100)
    # PROBE_LABEL="entropy" closes audit finding 4: Signal 3 must test whether
    # prompt-boundary activations predict multi-sample behavioral dispersion,
    # not supervised correctness. Ground-truth correctness enters only later,
    # as an external downstream benchmark on held-out data.
    PROBE_LABEL: str = "entropy"        # entropy (unsupervised) | correct (reference)
    PROBE_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    P0_NEG_CONTROL_TOL: float = 0.10    # |AUROC(p0) - 0.50| tolerance for the
                                        # constant-token negative control
    PROBE_MAX_ITER: int = 2000
    PROBE_STORE_DTYPE: str = "float32"  # PLAN §16 standing risk 1
    AUROC_GATE: float = 0.65            # Gate 3
    LABEL_SHUFFLE_REPEATS: int = 20     # null distribution size
    SURFACE_BASELINE: bool = True       # TF-IDF prompt-only control (PLAN §14.1)

    # ------------------------------------------------------- entropy ------
    # Audit fix 3c: a question with only k parsed samples must not default to
    # maximum behavioral confidence. Entropy is normalized against the TOTAL
    # requested sample count N, and questions with fewer than ENTROPY_MIN_VALID
    # valid parses get confidence NaN instead of a spuriously confident score.
    ENTROPY_MIN_VALID: int = 8          # of N_SAMPLES=10

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
    ONLY_MODELS=("qwen3.5-0.8b-instruct", "qwen3.5-2b-instruct"),
    COMMIT_CELLS_OUTSIDE_BAND=True,
) if SMOKE else _BASE_CFG

# --------------------------------------------------------------------------
# Other quick-start overrides — uncomment instead of editing the dataclass.
# --------------------------------------------------------------------------
# H0-only abort branch (PLAN §13 — publishable alone, no probe, no sampling):
# CFG = replace(CFG, RUN_NAME="h0_only",
#               STAGES=("data", "verbal", "grade", "stats", "figures", "tables", "report"))
#
# Session 1 of 2 — small models, leave the 9B pair for session 2:
# CFG = replace(CFG, RUN_NAME="s1", SKIP_MODELS=("qwen3.5-9b-instruct", "qwen3.5-9b-base"),
#               MODEL_EXEC="concurrent", MAX_CONCURRENT_MODELS=3)
#
# Session 2 of 2 — the 9B pair only (resumes shared question bank):
# CFG = replace(CFG, RUN_NAME="s1",
# CFG = replace(CFG, RUN_NAME="v35_full1000",
#               ONLY_MODELS=("qwen3.5-9b-instruct", "qwen3.5-9b-base"))

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
    """Bind every run to a real git commit (audit fix: `code_sha: "nogit"`
    made runs unreproducible). Walks up from the output dir looking for a
    .git anchor, because notebook platforms copy the pipeline file WITHOUT
    the repo. CODE_SHA env var overrides (set it when launching on Molab/
    Kaggle from a clone)."""
    env = os.environ.get("CODE_SHA", "").strip()
    if env:
        return env
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return r.stdout.strip()
        probe = Path.cwd()
        for _ in range(6):
            if (probe / ".git").exists():
                r = subprocess.run(["git", "-C", str(probe), "rev-parse", "--short", "HEAD"],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    return r.stdout.strip()
            if probe.parent == probe:
                break
            probe = probe.parent
    except Exception:
        pass
    try:                                                  # LOG may not exist yet on first call
        LOG.log("code_sha_missing", hint="no .git found — set CODE_SHA env or run inside the repo clone")
    except NameError:
        pass
    return "nogit"


def build_provenance(cfg: Config) -> dict:
    """PLAN §14.4 / X1 — stamped onto every derived artefact."""
    return {
        "run_name": cfg.RUN_NAME,
        "config_hash": cfg.hash(),
        "seed": cfg.SEED,
        "code_sha": code_sha(),
        "platform": cfg.resolved_platform(),
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),   # presence only, never the value
        "linear_attn_kernels": os.environ.get("LINEAR_ATTN_KERNELS", "unknown"),
        "devices": DEVICES,
        "dtype": str(cfg.resolved_dtype()),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "transformers": importlib.metadata.version("transformers"),
        "torch": torch.__version__,
        "math_verify": HAS_MATH_VERIFY,
    }


PROV = build_provenance(CFG)
(PATHS["meta"] / "provenance.json").write_text(json.dumps(PROV, indent=2, default=str), encoding="utf-8")
(PATHS["meta"] / "config.json").write_text(json.dumps(asdict(CFG), indent=2, default=str), encoding="utf-8")


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
    with path.open("r", encoding="utf-8", errors="replace") as fh:
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
    with path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=ensure_ascii, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())     # a checkpoint that is not on disk is not a checkpoint


def json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def json_read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
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


# --- consistent assistant-turn tail (audit requirement) ---------------------
# Every chat prompt must end on the SAME token so the activation tap's "last
# prompt token" is constant across models. Reasoning models (e.g.
# Qwen3.5-4B / 9B) ship a template that opens an UNCLOSED <think> block at the
# assistant turn; left alone the model spends its entire generation budget
# inside <think> and never emits the parseable ANSWER:/CONFIDENCE: lines.
# We (a) ask the template to disable thinking when it supports that flag, and
# (b) hard-normalise the tail to a closed empty-think block identical to the
# smaller models' native templates, so all chat prompts end byte-identically.
ASSISTANT_MARKER = "<|im_start|>assistant"
CANON_ASSISTANT_TAIL = "\n<think>\n\n</think>\n\n"


def think_off_kwargs(tokenizer) -> dict:
    """Return the template kwarg that disables reasoning, if the template
    supports it. Qwen3-style templates gate the <think> block on either
    `enable_thinking` or `thinking`; unknown kwargs are silently ignored by
    apply_chat_template, so we only pass a name the template actually uses."""
    tpl = getattr(tokenizer, "chat_template", "") or ""
    if "enable_thinking" in tpl:
        return {"enable_thinking": False}
    if "thinking" in tpl:
        return {"thinking": False}
    return {}


def normalize_assistant_tail(text: str) -> str:
    """Force a constant assistant-trigger tail across all chat models."""
    idx = text.rfind(ASSISTANT_MARKER)
    if idx == -1:
        return text
    return text[:idx + len(ASSISTANT_MARKER)] + CANON_ASSISTANT_TAIL


def build_prompt(variant: str, question: str, tier_spec: dict, model_spec: dict,
                 tokenizer, cfg: Config) -> str:
    """Return the fully-rendered prompt string for one (variant, question)."""
    instr = instruction(variant, tier_spec)
    if model_spec["chat"]:
        msgs = [
            {"role": "system", "content": "You are a precise assistant. You always reply in the exact requested format."},
            {"role": "user", "content": f"{instr}\nQuestion: {question}"},
        ]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **think_off_kwargs(tokenizer))
        return normalize_assistant_tail(text)

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
    if raw is None:
        return None
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%?", raw)
    if not m:
        return None
    v = float(m.group(1))
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402


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


def text_stack(model):
    """Resolve the TEXT decoder module for hooks / prefill across wrapper
    architectures. Qwen2.5-style CausalLM: model.model -> Qwen2Model (has
    .layers/.embed_tokens). Qwen 3.5: ForConditionalGeneration -> model.model
    is a multimodal container whose TEXT stack lives under .language_model.
    Raises with the submodule inventory when nothing matches — a loud error,
    never a silent wrong-layer tap."""
    m = getattr(model, "model", model)
    lm_stack = getattr(m, "language_model", None)
    if lm_stack is not None and hasattr(lm_stack, "layers"):
        return lm_stack
    if hasattr(m, "layers") and hasattr(m, "embed_tokens"):
        return m
    inner = getattr(lm_stack if lm_stack is not None else m, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    names = [n for n, _ in m.named_children()]
    raise RuntimeError(
        f"cannot locate text decoder stack (children: {names}). "
        f"Update text_stack() for this architecture before tapping activations.")


def load_model(name: str, cfg: Config) -> LoadedModel:
    spec = MODEL_SPECS[name]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], padding_side="left", trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kwargs = dict(
        torch_dtype=cfg.resolved_dtype(),
        attn_implementation=cfg.ATTN_IMPL,
        device_map=device_map(cfg),
        low_cpu_mem_usage=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], **load_kwargs)
    except Exception as exc:                               # noqa: BLE001
        # Multimodal wrappers (e.g. Qwen3_5ForConditionalGeneration) are not
        # registered under AutoModelForCausalLM in every transformers version.
        LOG.log("causal_load_failed", model=name, error=str(exc)[:160],
                note="retrying as image-text-to-text")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(spec["hf_id"], **load_kwargs)
    model.eval()
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception:
            pass

    embed = None
    if hasattr(model, "get_input_embeddings"):
        embed = model.get_input_embeddings()
    elif hasattr(getattr(model, "model", None), "embed_tokens"):
        embed = model.model.embed_tokens
    elif hasattr(getattr(getattr(model, "model", None), "language_model", None), "embed_tokens"):
        embed = model.model.language_model.embed_tokens

    head = None
    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
    elif hasattr(model, "lm_head"):
        head = model.lm_head
    elif hasattr(getattr(model, "model", None), "lm_head"):
        head = model.model.lm_head
    elif hasattr(getattr(getattr(model, "model", None), "language_model", None), "lm_head"):
        head = model.model.language_model.lm_head

    if head is not None and embed is not None:
        if getattr(model.config, "tie_word_embeddings", True):
            head.weight = embed.weight

    dt = cfg.resolved_dtype()
    if dt is not None:
        try:
            model.to(dtype=dt)
        except Exception:
            for p in model.parameters():
                if p.dtype != dt:
                    p.data = p.data.to(dtype=dt)
            for b in model.buffers():
                if b.dtype == torch.float32 and dt in (torch.bfloat16, torch.float16):
                    b.data = b.data.to(dtype=dt)

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


class ActivationTap:
    """Registers forward hooks on the requested depth points and captures the
    last-position hidden vector of each forward pass.

    Audit fix (finding 1): the hook buffer must reflect the LAST PROMPT token,
    not the last GENERATED token. `freeze()` is called right after the
    dedicated prompt-prefill forward pass so that autoregressive decoding
    inside generate() can no longer overwrite the captured activations."""

    def __init__(self, lm: LoadedModel, percentiles: Sequence[int]):
        self.lm = lm
        self.map = percentile_layers(lm.spec["layers"], percentiles)
        self.buffer: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self.frozen = False
        base = text_stack(lm.model)              # text decoder across wrappers
        for pct, idx in self.map.items():
            module = base.embed_tokens if idx == 0 else base.layers[idx - 1]
            self._handles.append(module.register_forward_hook(self._make_hook(pct)))

    def _make_hook(self, pct: int) -> Callable:
        def hook(_module, _inp, out):
            if self.frozen:
                return
            h = out[0] if isinstance(out, tuple) else out
            # left padding => index -1 is the true last prompt token for all rows
            self.buffer[pct] = h[:, -1, :].detach().to(torch.float32).cpu()
        return hook

    def pop(self) -> dict[int, np.ndarray]:
        out = {p: t.numpy() for p, t in self.buffer.items()}
        self.buffer = {}
        return out

    def freeze(self) -> None:
        """Stop capturing: decode-step forwards are ignored from here on."""
        self.frozen = True

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


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


@torch.no_grad()
def generate_batch(lm: LoadedModel, prompts: list[str], max_new_tokens: int,
                   temperature: float, top_p: float, n_return: int,
                   tap: "ActivationTap | None" = None,
                   want_logprobs: bool = False) -> tuple[list[list[str]], dict, list[list[float]]]:
    """Returns (texts grouped per prompt, activations, per-sequence
    length-normalized log-probs grouped per prompt).

    Activation capture (audit fix 1): a DEDICATED prefill forward pass over the
    prompt runs BEFORE generate(); the tap is popped immediately (last PROMPT
    token) and frozen so the autoregressive decode cannot overwrite it. The
    prefill goes through the transformer body only (no LM-head logits), so it
    costs one extra context pass and no vocab-sized tensor.
    """
    tok = lm.tokenizer
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536)
    enc = {k: v.to(lm.hidden_device) for k, v in enc.items()}
    do_sample = temperature and temperature > 0

    acts: dict[int, np.ndarray] = {}
    if tap is not None:
        base = text_stack(lm.model)              # transformer body: no logits head
        with torch.no_grad():
            base(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
        acts = tap.pop()
        tap.freeze()

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=bool(do_sample),
        num_return_sequences=n_return,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=False,
    )
    record_lp = bool(want_logprobs and do_sample)
    if do_sample:
        gen_kwargs.update(temperature=float(temperature), top_p=float(top_p))
    if record_lp:
        gen_kwargs.update(return_dict_in_generate=True, output_scores=True)
    out = lm.model.generate(**enc, **gen_kwargs)

    seq_lps: list[list[float]] = []
    if record_lp:
        # Per-token log-probs, computed MANUALLY rather than via
        # GenerateDecoderOnlyOutput.compute_transition_scores: that method was
        # removed from generation outputs in recent transformers (>=4.5x) and
        # its sampling-correct usage is normalize_logits=True anyway. This
        # reproduces exactly that: log_softmax over the processed scores,
        # gather the sampled token, pad-masked length-normalised mean.
        import torch.nn.functional as F
        seqs = out.sequences
        new_tokens = seqs[:, enc["input_ids"].shape[1]:]
        steps = []
        for step_score, tok_col in zip(out.scores, new_tokens.T):
            lp_step = F.log_softmax(step_score.float(), dim=-1)
            steps.append(lp_step.gather(1, tok_col.unsqueeze(-1)).squeeze(-1))
        tsc = torch.stack(steps, dim=-1)                      # [batch*n, gen_len]
        pad_mask = (new_tokens != tok.pad_token_id).float()
        lengths = pad_mask.sum(dim=-1).clamp(min=1)
        lp = ((tsc * pad_mask).sum(dim=-1) / lengths).cpu().numpy()
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        flat = [round(float(x), 6) for x in lp]
        seq_lps = [flat[i * n_return : (i + 1) * n_return] for i in range(len(prompts))]
    else:
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    grouped = [texts[i * n_return : (i + 1) * n_return] for i in range(len(prompts))]
    return grouped, acts, seq_lps


def run_generation(lm: LoadedModel, items: list[dict], variant: str, tier: str, cfg: Config,
                   ckpt: Checkpoint, n_return: int = 1, temperature: float | None = None,
                   capture_activations: bool = False,
                   prompt_variant: str | None = None) -> dict:
    """Drive one (model, tier, variant) pass with resume, OOM backoff, and
    periodic cache clearing. Returns a stats dict for the ledger.

    `prompt_variant` decouples the PROMPT template from the checkpoint key
    (audit fix: EXTRACT taps activations on exactly the SAMPLE prompts so the
    three signals are measured on the same generation context)."""
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
            prompts = [build_prompt(prompt_variant or variant, it["question"], tier_spec,
                                    lm.spec, lm.tokenizer, cfg)
                       for it in chunk]
            try:
                gens, acts, lps = generate_batch(lm, prompts, tier_spec["max_new_tokens"], temp,
                                                 cfg.SAMPLE_TOP_P, n_return, tap,
                                                 want_logprobs=bool(cfg.SAMPLE_SEQ_LOGPROBS))
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
                if cfg.SAVE_RAW_TEXT:
                    rec["raw"] = texts if n_return > 1 else texts[0]
                if n_return > 1 and lps:
                    rec["seq_logprob"] = lps[j] if j < len(lps) else []
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


def stage_extract(lm: LoadedModel, bank: dict, cfg: Config, committed: set[str]) -> dict:
    """Signal 3 — single greedy pass, five percentile taps at the last prompt
    token (PLAN §6). Uses the SAMPLE prompt so the probe reads a plain
    answering context, not a confidence-elicitation context.

    Audit fix (finding 4): `prompt_variant="SAMPLE"` renders the IDENTICAL
    prompt strings the behavioral signal samples from, so verbal/behavioral/
    internal are compared on aligned generation contexts rather than divergent
    trajectories. The checkpoint key stays "EXTRACT"."""
    out = {}
    for tier in cfg.active_tiers():
        if cell_id(lm.name, tier) not in committed:
            continue
        with open_ckpt("extract", lm.name, tier, cfg) as ck:
            out[tier] = run_generation(lm, cell_items(bank, tier, "all"), "EXTRACT", tier, cfg, ck,
                                       temperature=0.0, capture_activations=True,
                                       prompt_variant="SAMPLE")
    return out


def evaluate_band_gate(bank: dict, cfg: Config, nli: "NLIGrader | None") -> dict:
    """Grade the pilot to MEASURE per-cell accuracy (PLAN §3).

    Audit resolution (finding 9 / §5): cells are no longer DELETED for
    falling outside the 25–80% band — deletion discarded 75% of the grid
    (all of R2/R3, all of 0.5B, most of C3) and selection ran on a noisy
    n=100 pilot. Every cell is committed; pilot accuracy is recorded per
    cell and used downstream for within-cell matching and as a covariate
    (the base-rate CONTROL is kept, the deletion is dropped).
    COMMIT_CELLS_OUTSIDE_BAND is retained only for backwards compatibility
    and is ignored by this gate."""
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
            committed = True          # deletion filter REMOVED (audit finding 9)
            verdicts[cell_id(model, tier)] = {
                "model": model, "tier": tier, "n": n_tot, "accuracy": round(acc, 4),
                "band": [lo, hi], "in_band": in_band, "committed": committed,
                "parse_rate": round(float(np.mean([r["parse_ok"] for r in recs])), 4),
            }
    json_write(PATHS["derived"] / "cell_commitments.json", verdicts)
    n_c = sum(v["committed"] for v in verdicts.values())
    LOG.log("band_gate", cells=len(verdicts), committed=n_c,
            out_of_band_committed=len(verdicts) - sum(v["in_band"] for v in verdicts.values()),
            note="deletion filter disabled — accuracy recorded for matching/covariate use")
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


_CLUSTER_MATH_SUBS = [
    (r"\\left|\\right|\\!|\\,|\\;", ""),
    (r"\\dfrac|\\tfrac", r"\\frac"),
    (r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))"),
    (r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)"),
    (r"\^\s*\{([^{}]*)\}", r"^(\1)"),
    (r"\\(text|mbox|boxed)\{([^{}]*)\}", r"\2"),
]


def _cluster_math_norm(s: str) -> str:
    """LaTeX -> plain arithmetic for CLUSTERING (independent of math_verify,
    whose absence used to leave \\frac answers stranded as distinct clusters).
    '\\frac{11}{15}' -> '((11)/(15))', 'x = 60' -> 'x = 60'."""
    t = (s or "").strip().strip("$").strip()
    for pat, rep in _CLUSTER_MATH_SUBS:
        t = re.sub(pat, rep, t)
    t = t.replace("\\cdot", "*").replace("\\times", "*")
    t = t.replace(",", "").replace("%", "").replace("$", "")
    return t.strip().lower()


def _math_pair_equivalent(a: str, b: str) -> bool:
    """Bidirectional MATH equivalence between two answer strings for semantic
    clustering (audit fix 3b: '11/15' == '\\frac{11}{15}', '0.5' == '1/2',
    'x = 60' == '60'). Order: normalised identity -> pure-number tolerance ->
    sympy expression identity -> grading oracle."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    x, y = _cluster_math_norm(a), _cluster_math_norm(b)
    if x == y:
        return True
    # Pure numbers (no / or ^): compare with relative tolerance. extract_number
    # takes the LAST number, which is wrong for ratios like 11/15 — hence the
    # slash/exponent guard.
    if all(ch not in x + y for ch in "/^"):
        fx, fy = extract_number(x), extract_number(y)
        if fx is not None and fy is not None:
            return abs(fx - fy) <= 1e-6 * max(1.0, abs(fx), abs(fy))
    if SYMPY_OK:
        try:
            ex = sym_parse_expr(x.replace("^", "**"), evaluate=True)
            ey = sym_parse_expr(y.replace("^", "**"), evaluate=True)
            diff = ex - ey
            return bool(diff == 0 or (hasattr(diff, "equals") and diff.equals(0)))
        except Exception:                                  # noqa: BLE001
            pass
    try:
        return bool(grade_latex(a, [b]) or grade_latex(b, [a]))
    except Exception:                                      # noqa: BLE001
        return False


def cluster_answers(answers: list[str], answer_form: str, cfg: Config,
                     nli: "NLIGrader | None") -> list[int]:
    """Fast path: normalised string identity. Math forms: symbolic/numeric
    equivalence via the grading oracle, then union-find. Robust path:
    bidirectional NLI entailment + agglomerative merge (PLAN §5·2).

    Audit fix 3b: NLI merging used to be gated on `answer_form ∈ {short,
    entity}`, so every GSM8K/MATH cell silently degraded to exact-string
    clustering. The gate now includes math forms, and math answers are first
    merged by symbolic/numeric equivalence."""
    norm = [normalize_math(a) if answer_form == "latex" else normalize_text(a, cfg.STRIP_ARTICLES)
            for a in answers]
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

    originals: dict[int, str] = {}
    for a, l in zip(answers, labels):
        originals.setdefault(l, a)
    keys = sorted(originals)
    if len(keys) < 2:
        return labels

    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def compress(mapping: dict) -> list[int]:
        canon = {v: i for i, v in enumerate(sorted(set(mapping.values())))}
        return [canon[mapping[l]] for l in labels]

    # Math/symbolic merge — cheap oracle pairs before any NLI call.
    if answer_form in ("latex", "numeric"):
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if find(keys[i]) == find(keys[j]):
                    continue
                if _math_pair_equivalent(originals[keys[i]], originals[keys[j]]):
                    parent[find(keys[i])] = find(keys[j])
        remapped = {k: find(k) for k in keys}
        if len(set(remapped.values())) < len(keys):
            labels = compress(remapped)
            originals = {}
            for a, l in zip(answers, labels):
                originals.setdefault(l, a)
            keys = sorted(originals)
            parent = {k: k for k in keys}

    if not (cfg.USE_NLI_FALLBACK and nli is not None
            and answer_form in ("short", "entity", "latex", "numeric")):
        return labels
    if len(keys) < 2:
        return labels
    # Merge string-distinct clusters that entail each other both ways.
    pairs, meta = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pairs += [(originals[keys[i]], originals[keys[j]]), (originals[keys[j]], originals[keys[i]])]
            meta.append((keys[i], keys[j]))
    if not pairs:
        return labels
    try:
        s = nli.entails(pairs)
    except Exception:                                   # noqa: BLE001
        return labels

    for idx, (a, b) in enumerate(meta):
        if min(s[2 * idx], s[2 * idx + 1]) >= cfg.NLI_ENTAIL_THRESHOLD:
            parent[find(a)] = find(b)
    remap = {k: find(k) for k in keys}
    canon = {v: i for i, v in enumerate(sorted(set(remap.values())))}
    return [canon[remap[l]] for l in labels]


def stage_entropy(bank: dict, cfg: Config, nli: "NLIGrader | None") -> pd.DataFrame:
    """Shannon entropy over semantic cluster mass -> behavioral confidence.

    Audit fixes over the v2 shortcut implementation:
      * Rao-Blackwellization — cluster masses are weighted by
        length-normalized sequence log-probs recorded at sampling time
        (`seq_logprob`), replacing raw discrete sample counts; uniform
        weights are the documented fallback when log-probs are absent.
      * Normalization is against the TOTAL requested samples N (not the
        valid count), so parse failures cost confidence instead of
        inflating it.
      * Questions with fewer than ENTROPY_MIN_VALID parsed samples get
        confidence NaN rather than the old spurious 1.0 (n=1 case)."""
    rows = []
    N_total = int(cfg.N_SAMPLES)
    H_max = float(np.log(N_total))
    for model in cfg.active_models():
        for tier in cfg.active_tiers():
            recs = jsonl_read(raw_path("sample", model, tier))
            if not recs:
                continue
            gold = {r["qid"]: r for r in bank[tier]}
            for r in tqdm(recs, desc=f"entropy {model[:14]}|{tier}", leave=False):
                q = gold.get(r["qid"])
                if q is None:
                    continue
                parsed = r["parsed"] if isinstance(r["parsed"], list) else [r["parsed"]]
                lps = r.get("seq_logprob") or []
                pairs = [(p.get("answer"),
                          lps[i] if i < len(lps) and lps[i] is not None else None)
                         for i, p in enumerate(parsed)]
                valid = [(a, lp) for a, lp in pairs if a]
                n = len(valid)
                base_row = dict(model=model, tier=tier, qid=r["qid"], split=r["split"], n_valid=n)
                if n == 0:
                    rows.append({**base_row, "n_clusters": 0, "entropy": np.nan,
                                 "entropy_max": H_max, "confidence_behavioral": np.nan,
                                 "modal_share": np.nan, "low_valid": True})
                    continue
                labels = cluster_answers([a for a, _ in valid], q["answer_form"], cfg, nli)
                if all(lp is not None for _, lp in valid):
                    x = np.array([lp for _, lp in valid], dtype=float)
                    x = np.exp(x - x.max())
                    weights = x / x.sum()
                else:
                    weights = np.ones(n) / float(n)     # uniform = raw counts fallback
                masses = np.zeros(max(labels) + 1, dtype=float)
                for li, wi in zip(labels, weights):
                    masses[li] += wi
                H = float(-(masses * np.log(np.clip(masses, 1e-12, None))).sum())
                conf = float(1.0 - min(H / H_max, 1.0)) if H_max > 0 else np.nan
                low_valid = n < int(cfg.ENTROPY_MIN_VALID)
                rows.append({**base_row,
                             "n_clusters": int((masses > 1e-9).sum()),
                             "entropy": round(H, 6), "entropy_max": H_max,
                             "confidence_behavioral": (np.nan if low_valid else round(conf, 6)),
                             "modal_share": round(float(masses.max()), 6),
                             "low_valid": bool(low_valid)})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_parquet(PATHS["derived"] / "entropy.parquet", index=False)
    LOG.log("entropy", rows=len(df),
            low_valid=int(df["low_valid"].sum()) if len(df) else 0,
            mean_conf=round(float(df["confidence_behavioral"].mean()), 4) if len(df) else None)
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
                  entropy: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y, mask). Binary correctness or thresholded semantic entropy."""
    if cfg.PROBE_LABEL == "entropy" and len(entropy):
        sub = entropy[(entropy.model == model) & (entropy.tier == tier)].set_index("qid")
        vals = np.array([sub["confidence_behavioral"].get(q, np.nan) for q in qids], dtype=float)
        mask = np.isfinite(vals)
        med = np.nanmedian(vals) if mask.any() else 0.5
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


def select_probe_C(X_tr: np.ndarray, y_tr: np.ndarray, X_ca: np.ndarray, y_ca: np.ndarray,
                   cfg: Config, seed: int) -> tuple[float, Any]:
    """Audit fix (`PROBE_C_GRID` was dead code, C=1.0 hardcoded, train AUROC
    saturated at 1.000 in 51/65 rows): C is selected on the CALIBRATION split
    over cfg.PROBE_C_GRID — never on train AUROC, which rewards overfitting."""
    best_C, best_clf, best_auroc = None, None, -np.inf
    for c in cfg.PROBE_C_GRID:
        clf = fit_probe(X_tr, y_tr, cfg, seed, C=c)
        a = safe_auroc(y_ca, clf.predict_proba(X_ca)[:, 1])
        if np.isfinite(a) and a > best_auroc:
            best_C, best_clf, best_auroc = float(c), clf, a
    if best_clf is None:
        return 1.0, fit_probe(X_tr, y_tr, cfg, seed, C=1.0)
    LOG.log("probe_c_selected", C=best_C, auroc_cal=round(best_auroc, 4))
    return best_C, best_clf


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def stage_probe(bank: dict, graded: pd.DataFrame, entropy: pd.DataFrame,
                committed: dict, cfg: Config) -> pd.DataFrame:
    """One logistic probe per (cell × percentile). Layer selection happens on
    the CALIBRATION split only (PLAN §6·6, §14.2) — never train, never test."""
    rows = []
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

        # Gate 3 ENFORCEMENT (audit fix: it was computed but never blocked).
        # Non-finite activations anywhere in the cell abort downstream probe
        # fitting for that cell instead of poisoning the training set.
        fin = json_read(PATHS["acts"] / f"{model}__{tier}.finiteness.json", {})
        stored_worst = max((f.get("nonfinite_frac", 0.0) for f in fin.values()), default=0.0)
        runtime_worst = 0.0
        for X_chk in mats.values():
            finite_rows = np.isfinite(X_chk).all(axis=1)
            runtime_worst = max(runtime_worst, float(1.0 - finite_rows.mean()))
        if stored_worst > 0.0 or runtime_worst > 0.0:
            LOG.log("gate3_blocked", cell=cid, stored_nonfinite=round(stored_worst, 6),
                    runtime_nonfinite=round(runtime_worst, 6),
                    note="non-finite activations — probing blocked for this cell")
            continue

        y, mask = probe_labels(model, tier, qids, graded, entropy, cfg)
        splits = np.array([split_of[tier].get(q, "train") for q in qids])

        tr = mask & (splits == "train")
        ca = mask & (splits == "calibration")
        te = mask & (splits == "test")
        if tr.sum() < 20 or ca.sum() < 10:
            LOG.log("probe_skipped", cell=cid, n_train=int(tr.sum()), n_cal=int(ca.sum()))
            continue

        for pct in cfg.PERCENTILES:
            X = mats.get(pct)
            if X is None:
                continue
            tr_p, ca_p, te_p = tr, ca, te
            if tr_p.sum() < 20 or ca_p.sum() < 10 or len(np.unique(y[tr_p])) < 2:
                continue

            best_C, clf = select_probe_C(X[tr_p], y[tr_p], X[ca_p], y[ca_p], cfg, cfg.SEED)
            s_tr = clf.predict_proba(X[tr_p])[:, 1]
            s_ca = clf.predict_proba(X[ca_p])[:, 1]
            s_te = clf.predict_proba(X[te_p])[:, 1] if te_p.sum() else np.array([])

            # Label-shuffle null (PLAN §14.1) — refit at the selected C so the
            # null mirrors the actual procedure.
            rng = np.random.default_rng(cfg.SEED + pct)
            null = []
            for _ in range(cfg.LABEL_SHUFFLE_REPEATS):
                yp = rng.permutation(y[tr_p])
                if len(np.unique(yp)) < 2:
                    continue
                null.append(safe_auroc(y[ca_p], fit_probe(X[tr_p], yp, cfg, cfg.SEED,
                                                          C=best_C).predict_proba(X[ca_p])[:, 1]))
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
                C_selected=best_C,
                # Negative control (audit fix): p0 reads the embedding of a
                # CONSTANT prompt-ending token, so its AUROC must be ~0.50.
                # A large deviation means the tap is reading generated tokens.
                p0_neg_control_dev=(abs(auroc_cal - 0.5) if pct == 0 else float("nan")),
                nonfinite_frac=fin.get(f"p{pct}", {}).get("nonfinite_frac", 0.0),
                meets_gate=bool(np.isfinite(auroc_cal) and auroc_cal >= cfg.AUROC_GATE),
            ))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_parquet(PATHS["derived"] / "probe_sweep.parquet", index=False)
    LOG.log("probe_sweep", rows=len(df), cells=int(df["cell"].nunique()) if len(df) else 0)
    return df


def gate3_verdict(sweep: pd.DataFrame, cfg: Config) -> dict:
    """Per-cell: finiteness pre-check (ENFORCED upstream — blocked cells never
    reach the sweep), the constant-token negative control at p0, then >=1
    percentile with AUROC >= gate that also beats the shuffle null and the
    surface baseline."""
    out = {}
    for cell, g in sweep.groupby("cell") if len(sweep) else []:
        worst_finite = float(g["nonfinite_frac"].max())
        best = g.loc[g["auroc_cal"].idxmax()] if g["auroc_cal"].notna().any() else None
        p0 = g[(g["layer_pct"] == 0) & g["auroc_cal"].notna()]
        p0_dev = float(p0["auroc_cal"].sub(0.5).abs().max()) if len(p0) else float("nan")
        neg_control_ok = bool(not np.isfinite(p0_dev) or p0_dev <= cfg.P0_NEG_CONTROL_TOL)
        passes = bool(best is not None and best["meets_gate"] and best["beats_null"]
                      and (best["beats_surface"] or not cfg.SURFACE_BASELINE)
                      and neg_control_ok)
        out[cell] = {
            "activations_clean": worst_finite == 0.0, "worst_nonfinite_frac": worst_finite,
            "negative_control_p0_dev": None if not np.isfinite(p0_dev) else round(p0_dev, 4),
            "negative_control_ok": neg_control_ok,
            "best_layer_pct": int(best["layer_pct"]) if best is not None else None,
            "best_C": float(best["C_selected"]) if best is not None and "C_selected" in g.columns else None,
            "best_auroc_cal": float(best["auroc_cal"]) if best is not None else None,
            "beats_null": bool(best["beats_null"]) if best is not None else False,
            "beats_surface": bool(best["beats_surface"]) if best is not None else False,
            "passes": passes,
            "diagnosis": ("dirty activations" if worst_finite > 0
                          else "negative control failed — tap likely reading generated tokens"
                          if not neg_control_ok
                          else "clean activations, no signal" if not passes
                          else "pass"),
        }
    json_write(PATHS["derived"] / "gate3.json", out)
    LOG.log("gate3", cells=len(out), passed=sum(v["passes"] for v in out.values()),
            neg_control_failed=sum(1 for v in out.values() if not v["negative_control_ok"]))
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
    """Convert each format to a common 0–1 scale (PLAN §4 agreement check §2).

    Audit fix (finding 2): Format A's raw numeric stated confidence is the
    CANONICAL verbal axis — the model's own number, scaled to [0, 1]. Format B
    is label-derived (bucket -> empirical accuracy fit on calibration), so it
    is reported only as an auxiliary SUPERVISED accuracy baseline, never as
    what the model said."""
    rows = []
    g = graded[graded.variant.isin(["A", "B", "C"]) & (graded.sample_idx == 0)]
    for r in g.itertuples():
        cid = cell_id(r.model, r.tier)
        if r.variant == "A":
            # Raw stated confidence on its native 0–100 scale -> [0, 1].
            v = r.confidence / 100.0 if r.confidence is not None else np.nan
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
            v = float(np.clip(v, 0.0, 1.0))
        except (TypeError, ValueError):
            v = np.nan
        rows.append(dict(model=r.model, tier=r.tier, qid=r.qid, split=r.split,
                         is_agreement=r.is_agreement, variant=r.variant, verbal_raw=v,
                         correct=int(r.correct), decision=r.decision, bucket=r.bucket))
    return pd.DataFrame(rows)


def internal_scores(sweep: pd.DataFrame, bank: dict, graded: pd.DataFrame,
                     entropy: pd.DataFrame, committed: dict, cfg: Config) -> pd.DataFrame:
    """Refit the winning-percentile probe and emit per-question test scores.
    Winner selected on CALIBRATION only (PLAN §6·6)."""
    rows = []
    if not len(sweep):
        return pd.DataFrame(rows)
    split_of = {t: {r["qid"]: r["split"] for r in bank[t]} for t in bank}
    for cell, g in sweep.groupby("cell"):
        g = g[g["auroc_cal"].notna()]
        if not len(g):
            continue
        best = g.loc[g["auroc_cal"].idxmax()]
        model, tier, pct = best["model"], best["tier"], int(best["layer_pct"])
        win_C = float(best.get("C_selected", 1.0) or 1.0)
        loaded = load_activations(model, tier)
        if loaded is None:
            continue
        qids, mats = loaded
        X = mats.get(pct)
        if X is None:
            continue
        y, mask = probe_labels(model, tier, qids, graded, entropy, cfg)
        splits = np.array([split_of[tier].get(q, "train") for q in qids])
        fin = np.isfinite(X).all(axis=1)
        tr = mask & fin & (splits == "train")
        if tr.sum() < 20 or len(np.unique(y[tr])) < 2:
            continue
        clf = fit_probe(X[tr], y[tr], cfg, cfg.SEED, C=win_C)
        s_all = clf.predict_proba(X)[:, 1]
        for i, q in enumerate(qids):
            if not fin[i]:
                continue
            rows.append(dict(model=model, tier=tier, qid=q, split=splits[i],
                             internal_raw=float(s_all[i]), best_layer_pct=pct))
    return pd.DataFrame(rows)


def assemble_signals(bank: dict, graded: pd.DataFrame, entropy: pd.DataFrame,
                     sweep: pd.DataFrame, committed: dict, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Wide table: one row per (model, tier, qid) carrying raw + calibrated
    values for all three signals, plus ground truth. Calibrators fit on the
    calibration split, applied to test (PLAN §8·1–2)."""
    bmap = empirical_bucket_map(graded, cfg)
    verbal = verbal_scores(graded, bmap, cfg)
    internal = internal_scores(sweep, bank, graded, entropy, committed, cfg)

    # Canonical verbal axis (audit fix, finding 2): Format A — the model's raw
    # stated confidence scaled to [0, 1] — is canonical BY CONSTRUCTION.
    # Format B is LABEL-DERIVED (each bucket maps to that bucket's empirical
    # accuracy fit on the calibration split, then calibrated again on the same
    # split: `format_stats.B.ece = 2.3e-17`, identically zero by
    # construction). B is therefore NEVER eligible for the canonical role; it
    # stays in fmt_stats / verbal_long as a supervised P(correct) reference,
    # reported separately.
    #
    # The §8·6 pre-flight (>= MIN_DISTINCT_VERBAL distinct values) still
    # applies to A before it can represent a cell, and to C as fallback.
    fmt_stats = {}
    for v in ("A", "B", "C"):
        s = verbal[(verbal.variant == v) & (verbal.split == "calibration")]
        s = s[s.verbal_raw.notna()]
        nd = int(s.verbal_raw.nunique()) if len(s) else 0
        ok = len(s) >= 20
        fmt_stats[v] = {
            "n": int(len(s)), "n_distinct": nd,
            "ece": ece(s.verbal_raw.values, s.correct.values, cfg.ECE_BINS) if ok else float("inf"),
            "brier": brier(s.verbal_raw.values, s.correct.values.astype(float)) if ok else float("inf"),
            "eligible": bool(ok and nd >= cfg.MIN_DISTINCT_VERBAL),
        }
    eligible = [v for v in ("A", "C") if fmt_stats[v]["eligible"]]   # B excluded by design
    if eligible:
        canonical = "A" if "A" in eligible else min(eligible, key=lambda v: fmt_stats[v]["brier"])
    else:
        # Nothing clears the pre-flight: keep the most-varied NON-label-derived
        # format so the pipeline still runs, but the degeneracy is recorded loudly.
        non_b = {v: fmt_stats[v] for v in ("A", "C")}
        canonical = max(non_b, key=lambda v: non_b[v]["n_distinct"])
        LOG.log("verbal_all_formats_degenerate", chosen=canonical,
                n_distinct={k: v["n_distinct"] for k, v in fmt_stats.items()})
    fmt_ece = {k: v["ece"] for k, v in fmt_stats.items()}
    LOG.log("canonical_format", chosen=canonical, eligible=eligible,
            note="B demoted to supervised accuracy baseline (label-derived); "
                 "canonical verbal axis = model-stated confidence",
            brier={k: round(v["brier"], 4) for k, v in fmt_stats.items()},
            ece={k: round(v["ece"], 4) for k, v in fmt_stats.items()},
            n_distinct={k: v["n_distinct"] for k, v in fmt_stats.items()})

    v_can = verbal[verbal.variant == canonical][["model", "tier", "qid", "split", "verbal_raw", "correct"]]
    base = v_can.copy()
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
    LOG.log("verbal_preflight", excluded_cells=sum(
        1 for m in cal_meta.values() if m.get("verbal", {}).get("excluded")),
        total_cells=len(cal_meta), excluded_rows=n_excl)

    base["family"] = base["tier"].map(lambda t: TIER_SPECS[t]["family"])
    base["params_b"] = base["model"].map(lambda m: MODEL_SPECS[m]["params_b"])
    base["canonical_format"] = canonical
    base.to_parquet(PATHS["derived"] / "signals.parquet", index=False)
    json_write(PATHS["derived"] / "calibration_meta.json",
               {"canonical_format": canonical, "format_ece": fmt_ece,
                "format_stats": fmt_stats, "cells": cal_meta})
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
        out["pooled"][sig] = {"ece": e, "ece_lo": elo, "ece_hi": ehi,
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


QUADRANT_TAXONOMY = {
    # calibrated verbal vs mean(behavioral, internal), thresholded at 0.5
    ("high", "low"): "performative_certainty",   # legacy: "hopeful" — stated confidence
                                                 # exceeds sample stability
    ("low", "high"): "excessive_hedging",        # legacy: "suppressed" — stated confidence
                                                 # under-reports sample consensus
    ("high", "high"): "grounded_certainty",
    ("low", "low"): "honest_doubt",
}
QUADRANT_LEGACY = {"performative_certainty": "hopeful", "excessive_hedging": "suppressed",
                   "grounded_certainty": "agree_high", "honest_doubt": "agree_low"}


def test_h2_quadrants(signals: pd.DataFrame, bank: dict, cfg: Config,
                      gate1_pass: bool | None = None,
                      gate3: dict | None = None) -> dict:
    """Epistemic-alignment quadrant assignment (renamed per audit: retire the
    anthropomorphic 'hopeful/suppressed' labels), a CONTINUOUS discrepancy
    metric Δ = verbal_cal − behavioral_cal that does not depend on the 0.5
    cut, feature association tested globally AND required to replicate across
    multiple cells, and the PLAN §16 gate conjuncts RESTORED: H2 passes only
    if features associate AND Gate 1 holds AND Gate 3 activations were clean."""
    test = signals[signals.split == "test"].copy()
    thr = cfg.QUADRANT_THRESHOLD
    other = test[["behavioral_cal", "internal_cal"]].mean(axis=1, skipna=True)
    test["other_cal"] = other

    def quad(row):
        if not np.isfinite(row["verbal_cal"]) or not np.isfinite(row["other_cal"]):
            return None
        return QUADRANT_TAXONOMY[("high" if row["verbal_cal"] >= thr else "low",
                                  "high" if row["other_cal"] >= thr else "low")]
    test["quadrant"] = test.apply(quad, axis=1)
    test["quadrant_legacy"] = test["quadrant"].map(lambda q: QUADRANT_LEGACY.get(q))
    test["delta_verbal_minus_behavioral"] = test["verbal_cal"] - test["behavioral_cal"]
    feats = question_features(bank)
    merged = test.merge(feats, on=["tier", "qid"], how="left", suffixes=("", "_f"))
    merged = merged[merged.quadrant.notna()]

    counts = {k: int(v) for k, v in Counter(merged["quadrant"]).items()}
    out = {"counts": counts,
           "counts_legacy": {QUADRANT_LEGACY[k]: v for k, v in counts.items()},
           "n": int(len(merged)), "threshold": thr, "associations": {}}
    d = merged["delta_verbal_minus_behavioral"].dropna()
    out["delta_metric"] = {
        "n": int(len(d)), "mean": float(d.mean()) if len(d) else float("nan"),
        "median": float(d.median()) if len(d) else float("nan"),
        "note": "Δ = verbal_cal − behavioral_cal; continuous, independent of the 0.5 cut",
    }

    rng = np.random.default_rng(cfg.SEED)
    cell_ids = merged["model"] + "__" + merged["tier"]
    for feat in ("has_year", "is_long", "has_multi_entity", "has_number", "family"):
        if feat not in merged:
            continue
        tab = pd.crosstab(merged["quadrant"], merged[feat])
        if tab.shape[0] < 2 or tab.shape[1] < 2 or tab.values.sum() < 20:
            continue
        chi2, p, dof, _ = sps.chi2_contingency(tab)
        nulls = []
        for _ in range(200):
            shuffled = rng.permutation(merged[feat].values)
            t2 = pd.crosstab(merged["quadrant"], shuffled)
            if t2.shape[0] >= 2 and t2.shape[1] >= 2:
                nulls.append(sps.chi2_contingency(t2)[0])
        # Within-cell replication requirement (audit fix: H2's only passing
        # features were tier proxies carried by three cells). A general claim
        # about question characteristics needs the association to appear in
        # MORE THAN ONE cell.
        cells_sig = 0
        for _, gc in merged.groupby(cell_ids):
            tc = pd.crosstab(gc["quadrant"], gc[feat])
            if tc.shape[0] >= 2 and tc.shape[1] >= 2 and len(gc) >= 40:
                if sps.chi2_contingency(tc)[1] < 0.05:
                    cells_sig += 1
        out["associations"][feat] = {
            "chi2": float(chi2), "p": float(p), "dof": int(dof),
            "null_p95": float(np.percentile(nulls, 95)) if nulls else float("nan"),
            "beats_null": bool(nulls and chi2 > np.percentile(nulls, 95)),
            "cells_significant": int(cells_sig),
            "cells_tested": int(sum(1 for _, gc in merged.groupby(cell_ids)
                                    if len(gc) >= 40)),
            "replicates_across_cells": bool(cells_sig >= 2),
            "table": tab.to_dict(),
        }
    sig = [f for f, v in out["associations"].items()
           if v["p"] < 0.05 and v["beats_null"] and v["replicates_across_cells"]]
    assoc_ok = bool(sig)

    gate3_clean = bool(gate3) and all(bool(v.get("activations_clean")) for v in gate3.values())
    out["features_associate_and_replicate"] = assoc_ok
    out["gate1_pass"] = None if gate1_pass is None else bool(gate1_pass)
    out["gate3_activations_clean"] = gate3_clean
    # PLAN §16 conjunct restored (audit fix: it was silently dropped).
    out["h2_pass"] = bool(assoc_ok and gate1_pass is True and gate3_clean)
    failed = []
    if not assoc_ok:
        failed.append("no feature association replicates across cells")
    if gate1_pass is not True:
        failed.append("Gate 1 (grading sanity) does not hold")
    if not gate3:
        failed.append("Gate 3 not evaluated (no cells were probed)")
    elif not gate3_clean:
        failed.append("Gate 3 activations not clean")
    out["verdict"] = ("H2 supported — replicated feature association with the alignment taxonomy, "
                      "with Gate 1 and Gate 3 holding"
                      if out["h2_pass"] else
                      "H2 falsified / not claimable — " + "; ".join(failed))

    examples = {}
    qtext = {r["qid"]: r["question"] for t in bank for r in bank[t]}
    for q in ("performative_certainty", "excessive_hedging"):
        sel = merged[merged.quadrant == q].head(25)
        examples[q] = [{"model": r.model, "tier": r.tier, "qid": r.qid,
                        "question": qtext.get(r.qid, ""), "verbal": round(float(r.verbal_cal), 3),
                        "other": round(float(r.other_cal), 3), "correct": int(r.correct)}
                       for r in sel.itertuples()]
    out["examples"] = examples
    merged.to_parquet(PATHS["derived"] / "quadrants.parquet", index=False)
    json_write(PATHS["derived"] / "h2_quadrants.json", out)
    LOG.log("h2", pass_=out["h2_pass"], **out["counts"])
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


def h3_model_pair() -> tuple[str | None, str | None]:
    """Discover the (base, instruct) comparison pair from MODEL_SPECS rung
    tags instead of hardcoding qwen2.5 names — the ladder is now Qwen 3.5."""
    bases = [m for m, s in MODEL_SPECS.items() if s.get("rung") == "h3-comparison"]
    best = None
    for b in bases:
        stem = b.rsplit("-", 1)[0]
        inst = [m for m, s in MODEL_SPECS.items()
                if m.startswith(stem) and s.get("chat") and s.get("rung") == "ladder"]
        if not inst:
            continue
        i = max(inst, key=lambda m: MODEL_SPECS[m]["params_b"])
        if best is None or MODEL_SPECS[i]["params_b"] > MODEL_SPECS[best[1]]["params_b"]:
            best = (b, i)
    return best if best else (None, None)


def test_h3_base_vs_instruct(signals: pd.DataFrame, abst: dict, cfg: Config,
                             gate4_elicit_ok: bool | None = None) -> dict:
    """PLAN §13 H3 / Gate 4 — stated-confidence overstatement delta between the
    base and instruction-tuned models, with a missed-knowledge guard.

    Audit fixes:
      * NaN coercion bug: `((v>=thr)&(o<thr)).astype(float)` silently turned
        rows with missing verbal/other scores into artificial 0.0 (comparisons
        on NaN are False), which is what manufactured hopeful_rate_base = 0.0.
        Rows without the required scores are now excluded BEFORE the indicator
        is built and never enter the rates.
      * `hopeful` is NOT equated with 'confident and wrong': its correctness
        rate and the confident-incorrect rate are reported separately.
      * Terminology: outputs use the neutral taxonomy label 'unwarranted
        stated confidence' (Performative Certainty); legacy keys are kept as
        deprecated aliases so old readers don't break."""
    a, b = h3_model_pair()
    if a is None or b is None:
        r = {"available": False, "verdict": "H3 untested — no base/instruct pair in MODEL_SPECS"}
        json_write(PATHS["derived"] / "h3_model_delta.json", r)
        return r
    test = signals[(signals.split == "test") & signals.model.isin([a, b])].copy()
    if test.model.nunique() < 2:
        r = {"available": False, "verdict": f"H3 untested — both {a} and {b} required"}
        json_write(PATHS["derived"] / "h3_model_delta.json", r)
        return r
    thr = cfg.QUADRANT_THRESHOLD
    test["other_cal"] = test[["behavioral_cal", "internal_cal"]].mean(axis=1, skipna=True)
    # Exclude rows missing verbal OR all other-signal scores BEFORE building
    # the binary indicator; NaN-safe via np.where (never coerce to 0.0).
    valid = test["verbal_cal"].notna() & test["other_cal"].notna()
    test["unwarranted"] = np.where(
        valid, ((test["verbal_cal"] >= thr) & (test["other_cal"] < thr)).astype(float), np.nan)
    shared = set(test[test.model == a].qid) & set(test[test.model == b].qid)
    test = test[test.qid.isin(shared)]
    xa = test[test.model == a]["unwarranted"].dropna().values
    xb = test[test.model == b]["unwarranted"].dropna().values
    delta = bootstrap_diff_ci(xa, xb, np.mean, cfg.N_BOOTSTRAP, cfg.BOOTSTRAP_CI, cfg.SEED)

    def _rate(g: pd.DataFrame, col: str) -> float:
        d = g[col].dropna()
        return float(d.mean()) if len(d) else float("nan")

    corr_rates, conf_wrong, hop_corr = {}, {}, {}
    for name, m in (("base", a), ("instruct", b)):
        gm = test[test.model == m]
        corr_rates[name] = _rate(gm, "correct")
        vw = gm[(gm.verbal_cal >= thr)]
        conf_wrong[name] = float((vw["correct"] == 0).mean()) if len(vw) else float("nan")
        hu = gm[gm["unwarranted"] == 1.0]
        hop_corr[name] = float(hu["correct"].mean()) if len(hu) else float("nan")

    mk = {m: np.mean([v["missed_knowledge_rate"] for k, v in abst.get("per_cell", {}).items()
                      if k.startswith(m)] or [np.nan]) for m in (a, b)}
    mk_rise = float(mk[b] - mk[a]) if all(np.isfinite(list(mk.values()))) else float("nan")
    guard_ok = bool(not np.isfinite(mk_rise) or mk_rise < delta["delta"])
    out = {"available": True, "model_base": a, "model_instruct": b,
           "n_matched": int(len(shared)),
           "n_scored": {"base": int(xa.size), "instruct": int(xb.size)},
           "excluded_missing_scores": int((~valid).sum()),
           "unwarranted_confidence_rate": {"base": float(np.mean(xa)) if xa.size else float("nan"),
                                           "instruct": float(np.mean(xb)) if xb.size else float("nan")},
           "delta_base_minus_instruct": delta,
           # Correctness decomposition — overstatement ≠ being wrong.
           "hopeful_rows_correct_rate": {k: (None if not np.isfinite(v) else round(v, 4))
                                         for k, v in hop_corr.items()},
           "confident_incorrect_rate": {k: (None if not np.isfinite(v) else round(v, 4))
                                        for k, v in conf_wrong.items()},
           "overall_correct_rate": {k: (None if not np.isfinite(v) else round(v, 4))
                                    for k, v in corr_rates.items()},
           "missed_knowledge_rate": {k: (None if not np.isfinite(v) else float(v)) for k, v in mk.items()},
           "missed_knowledge_rise": mk_rise, "guard_passes": guard_ok}
    replicates = bool(delta["lo"] > 0 and guard_ok
                      and (gate4_elicit_ok is None or gate4_elicit_ok))
    out["guard_note"] = ("If this delta replicates after the NaN repair it should be framed as a "
                         "descriptive trade-off of instruction tuning in THIS evaluation setting, "
                         "not as a general 'hopeful confidence' claim.")
    out["h3_pass"] = replicates
    out["verdict"] = ("H3 supported — instruction tuning lowers unwarranted stated confidence "
                      "without blanket hedging"
                      if replicates else
                      "H3 falsified / null — delta CI includes 0, the missed-knowledge guard fired, "
                      "or base-model elicitation did not meet the usability bar")
    # Deprecated aliases (v2 readers).
    out["hopeful_rate_base"] = out["unwarranted_confidence_rate"]["base"]
    out["hopeful_rate_instruct"] = out["unwarranted_confidence_rate"]["instruct"]
    json_write(PATHS["derived"] / "h3_model_delta.json", out)
    LOG.log("h3", pass_=out["h3_pass"], delta=round(delta["delta"], 4),
            n_scored=f"{xa.size}/{xb.size}")
    return out


def test_h4_depth(sweep: pd.DataFrame, cfg: Config) -> dict:
    """PLAN §13 H4 — onset percentile (first percentile reaching AUROC >= gate)
    for retrieval vs reasoning, with a bootstrap CI on the difference."""
    if not len(sweep):
        return {"available": False}
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
    json_write(PATHS["derived"] / "h4_depth.json", out)
    LOG.log("h4", pass_=out["h4_pass"], ret=out["mean_onset_retrieval"], rea=out["mean_onset_reasoning"])
    return out


def hierarchical_regression(signals: pd.DataFrame, cfg: Config,
                            feats: pd.DataFrame | None = None) -> dict:
    """PLAN §8·4 — ONE pooled model across the grid, not 30 per-cell fits.
    Audit upgrade (finding 5 / Priority-3 #10): questions nested in tiers and
    models via random intercepts for question, model, tier AND model×tier, so
    cell baselines are absorbed instead of cells being deleted. Question
    characteristics enter as FIXED effects when a feature table is supplied."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = signals[signals.split == "test"].copy()
    d = d.dropna(subset=["verbal_cal", "correct"])
    for c in ("behavioral_cal", "internal_cal"):
        d[c] = d[c].fillna(d[c].mean())
    if feats is not None and len(feats):
        d = d.merge(feats.drop(columns=["family"], errors="ignore"),
                    on=["tier", "qid"], how="left")
        for c in ("is_long", "has_number", "has_multi_entity"):
            if c in d:
                d[c] = d[c].fillna(0).astype(int)
    if len(d) < 50:
        return {"available": False, "n": int(len(d)), "reason": "insufficient test rows"}
    d["log_params"] = np.log(d["params_b"].clip(lower=0.01))
    d["layer_pct"] = d.get("best_layer_pct", pd.Series(np.nan, index=d.index)).fillna(50.0)
    d["is_reasoning"] = (d["family"] == "reasoning").astype(int)
    d["model_x_tier"] = d["model"] + "__" + d["tier"]
    fixed = ("correct ~ verbal_cal + behavioral_cal + internal_cal + is_reasoning + log_params"
             " + is_reasoning:internal_cal + log_params:internal_cal + layer_pct:is_reasoning")
    if {"is_long", "has_number"}.issubset(d.columns):
        fixed += " + is_long + has_number"
    formula = fixed
    out: dict[str, Any] = {"available": True, "n": int(len(d)), "formula": formula}
    method = cfg.HLR_METHOD
    if method in ("auto", "bayes_mixed"):
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            vocs = {"question": "0 + C(qid)", "model": "0 + C(model)",
                    "tier": "0 + C(tier)", "cell": "0 + C(model_x_tier)"}
            m = BinomialBayesMixedGLM.from_formula(formula, vocs, d)
            r = m.fit_vb(verbose=False)
            out.update(method="bayes_mixed_glm",
                       random_effects=list(vocs),
                       params=dict(zip(r.model.exog_names, [float(x) for x in r.fe_mean])),
                       sd=dict(zip(r.model.exog_names, [float(x) for x in r.fe_sd])))
            json_write(PATHS["derived"] / "hierarchical_regression.json", out)
            return out
        except Exception as exc:                          # noqa: BLE001
            out["bayes_error"] = str(exc)[:200]
    try:
        m = smf.logit(formula, data=d).fit(disp=0, cov_type="cluster",
                                           cov_kwds={"groups": d["model_x_tier"]})
        out.update(method="logit_cluster_robust",
                   params={k: float(v) for k, v in m.params.items()},
                   pvalues={k: float(v) for k, v in m.pvalues.items()},
                   conf_int={k: [float(a), float(b)] for k, (a, b) in m.conf_int().iterrows()},
                   pseudo_r2=float(m.prsquared), summary=str(m.summary()))
    except Exception as exc:                              # noqa: BLE001
        out.update(method="failed", error=str(exc)[:300])
    json_write(PATHS["derived"] / "hierarchical_regression.json", out)
    LOG.log("hlr", method=out.get("method"), n=out["n"])
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


def popularity_contrast(signals: pd.DataFrame, cfg: Config) -> dict:
    """Audit recovery task: the popularity manipulation (R1 PopQA-high vs
    R2 PopQA-low) is the design's most distinctive feature and was never
    analyzed while R2 was deleted by the band filter. With all cells committed
    this contrasts per-signal raw scores and correctness between R1 and R2
    within each model, on matched test rows."""
    out = {"per_model": {}}
    test = signals[signals.split == "test"]
    tiers = [t for t in ("R1", "R2") if t in set(test["tier"])]
    if len(tiers) < 2:
        return {"available": False, "reason": "need both R1 and R2 committed"}
    for model, g in test.groupby("model"):
        hi = g[g.tier == "R1"]
        lo = g[g.tier == "R2"]
        entry = {"n": {"R1": int(len(hi)), "R2": int(len(lo))}}
        for sig in SIGNALS:
            a = hi[f"{sig}_raw"].dropna()
            b = lo[f"{sig}_raw"].dropna()
            if len(a) > 5 and len(b) > 5:
                sp = spearman_with_ci(
                    pd.concat([a, b]).values,
                    np.concatenate([np.ones(len(a)), np.zeros(len(b))]),
                    min(cfg.N_BOOTSTRAP, 500), cfg.BOOTSTRAP_CI, cfg.SEED)
                entry[sig] = {"mean_R1": float(a.mean()), "mean_R2": float(b.mean()),
                              "diff_auc_rho": round(sp["rho"], 4)}
        ca, cb = hi["correct"].astype(float), lo["correct"].astype(float)
        entry["accuracy"] = {"R1": round(float(ca.mean()), 4), "R2": round(float(cb.mean()), 4)}
        out["per_model"][model] = entry
    out["available"] = True
    json_write(PATHS["derived"] / "popularity_gradient.json", out)
    LOG.log("popularity_gradient", models=len(out["per_model"]))
    return out


def difficulty_matched_view(signals: pd.DataFrame, graded: pd.DataFrame,
                            cfg: Config, n_bins: int = 4) -> dict:
    """Base-rate control BY MATCHING instead of deletion (audit fix §9).

    Questions are stratified into difficulty bins from their cross-model mean
    FORCED-answer accuracy within tier; per-bin ECE/resolution are then
    computed per signal so cross-signal comparisons contrast models on the
    SAME difficulty profile rather than on different accuracy mixes."""
    forced = graded[(graded.variant == "FORCED") & (graded.split == "test")]
    if not len(forced):
        return {"available": False, "reason": "no FORCED rows"}
    diff = (forced.groupby(["tier", "qid"])["correct"].mean()
            .rename("difficulty").reset_index())
    diff["difficulty_bin"] = diff.groupby("tier")["difficulty"].transform(
        lambda s: pd.qcut(s, q=min(n_bins, max(2, s.nunique())), labels=False,
                          duplicates="drop"))
    d = signals[signals.split == "test"].merge(diff[["tier", "qid", "difficulty_bin"]],
                                               on=["tier", "qid"], how="inner")
    bins = sorted(x for x in d["difficulty_bin"].unique() if pd.notna(x))
    out = {"n_bins": len(bins), "bins": {}}
    for b in bins:
        db = d[d.difficulty_bin == b]
        entry = {"n": int(len(db)),
                 "accuracy": round(float(db["correct"].astype(float).mean()), 4)}
        for sig in SIGNALS:
            dd = db[[f"{sig}_cal", "correct"]].dropna()
            if len(dd) < 20:
                continue
            p, y = dd[f"{sig}_cal"].values, dd["correct"].values.astype(float)
            entry[sig] = {"ece": ece(p, y, cfg.ECE_BINS),
                          **{k: round(v, 5) for k, v in murphy_decomposition(
                              p, y, cfg.MURPHY_BINS).items() if k in ("resolution", "reliability")}}
        out["bins"][int(b)] = entry
    out["available"] = bool(any("verbal" in v or "behavioral" in v or "internal" in v
                                for v in out["bins"].values()))
    json_write(PATHS["derived"] / "difficulty_matched.json", out)
    LOG.log("difficulty_matched", bins=out["n_bins"], available=out["available"])
    return out
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


JUDGE = JudgeConfig(ENABLED=True)

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
    """Runs after everything else. Idempotent and resumable like every stage."""
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

    Audit fix (finding 8): a fresh run used to clobber the HUMAN-ANNOTATED
    sheet with a blank template (manual_correct was reset to empty in every
    result tree). If the sheet on disk already carries labels, it is preserved
    untouched and re-ingested into the final report instead."""
    p = PATHS["tables"] / "gate1_manual_check_sheet.csv"
    if p.exists():
        try:
            old = pd.read_csv(p)
            if "manual_correct" in old.columns and old["manual_correct"].notna().any():
                LOG.log("manual_sheet_preserved", path=str(p),
                        rows=int(len(old)),
                        filled=int(old["manual_correct"].notna().sum()))
                return p
        except Exception:                                  # noqa: BLE001
            pass
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
    df.to_csv(p, index=False)
    LOG.log("manual_sheet", path=str(p), rows=len(df))
    return p


def ingest_manual_sheet(cfg: Config) -> dict:
    """Read back a human-filled Gate 1 check sheet and compute the agreement
    statistics that close Gate 1 (audit fix: the filled sheet was never
    ingested; PLAN §16 requires the manual agreement rate in the report)."""
    p = PATHS["tables"] / "gate1_manual_check_sheet.csv"
    if not p.exists():
        return {"available": False, "reason": "sheet not found"}
    try:
        df = pd.read_csv(p)
    except Exception as exc:                               # noqa: BLE001
        return {"available": False, "reason": str(exc)[:120]}
    if "manual_correct" not in df.columns or not df["manual_correct"].notna().any():
        return {"available": False, "reason": "no manual labels yet",
                "path": str(p)}
    d = df[df["manual_correct"].notna()].copy()
    d["manual_correct"] = d["manual_correct"].astype(str).str.strip().str.lower() \
        .map({"true": True, "false": False, "1": True, "0": False})
    d = d[d["manual_correct"].notna()]
    agree = float((d["manual_correct"] == d["automated_correct"].astype(bool)).mean()) \
        if len(d) else float("nan")
    by_grader = {str(k): round(float((g["manual_correct"] == g["automated_correct"].astype(bool)).mean()), 4)
                 for k, g in d.groupby("grader")} if len(d) else {}
    res = {
        "available": True, "n_filled": int(len(d)), "n_total": int(len(df)),
        "agreement": round(agree, 4) if np.isfinite(agree) else None,
        "agreement_by_grader": by_grader,
        "gate1_threshold": cfg.GATE1_AGREEMENT,
        "gate1_manual_pass": bool(np.isfinite(agree) and agree >= cfg.GATE1_AGREEMENT),
        "disagreements": int((d["manual_correct"] != d["automated_correct"].astype(bool)).sum()),
    }
    json_write(PATHS["derived"] / "gate1_manual.json", res)
    LOG.log("gate1_manual", n=res["n_filled"], agreement=res["agreement"],
            pass_=res["gate1_manual_pass"])
    return res
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
QUADRANT_COLOR = {"performative_certainty": PALETTE[0], "excessive_hedging": PALETTE[1],
                  "grounded_certainty": "#c9c8c2", "honest_doubt": "#e6e5e1",
                  # legacy names kept so old quadrant parquets still render
                  "hopeful": PALETTE[0], "suppressed": PALETTE[1],
                  "agree_high": "#c9c8c2", "agree_low": "#e6e5e1"}


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


def reliability_curve(p: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() < 3:
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
    ax.legend(loc="upper left")

    ax2 = axes[1]
    comps, labels = ["reliability", "resolution"], []
    width = 0.36
    for k, sig in enumerate(SIGNALS):
        v = h1.get("pooled", {}).get(sig, {})
        if "reliability" not in v:
            continue
        labels.append(sig)
        for j, comp in enumerate(comps):
            ax2.bar(len(labels) - 1 + (j - 0.5) * width, v.get(comp, np.nan), width * 0.92,
                    color=SIGNAL_COLOR[sig], alpha=1.0 if j == 0 else 0.45,
                    edgecolor=SURFACE, linewidth=1.2)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set(ylabel="Brier component", title="Murphy decomposition")
    ax2.legend(handles=[Line2D([], [], color=INK3, lw=6, alpha=1.0, label="reliability (lower better)"),
                        Line2D([], [], color=INK3, lw=6, alpha=0.45, label="resolution (higher better)")],
              loc="upper right")
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
    ax.axvline(thr, color=GRID, lw=1.0, zorder=1)
    ax.axhline(thr, color=GRID, lw=1.0, zorder=1)
    for name in ("honest_doubt", "grounded_certainty", "excessive_hedging",
                 "performative_certainty"):   # mismatches drawn on top
        s = q[q.quadrant == name]
        if not len(s):
            continue
        ax.scatter(s["verbal_cal"], s["other_cal"], s=13, color=QUADRANT_COLOR[name],
                   edgecolors=SURFACE, linewidths=0.5, alpha=0.85,
                   zorder=3 if name in ("performative_certainty", "excessive_hedging") else 2,
                   label=f"{name.replace('_', ' ')} (n={len(s)})")
    ax.set(xlabel="calibrated verbalized confidence", ylabel="calibrated behavioral / internal",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
           title="Epistemic alignment quadrants (test split)")
    ax.legend(loc="lower right", markerscale=1.4)

    ax2 = axes[1]
    cats = ["justified_hedge", "missed_knowledge", "unresolved"]
    vals = [abst.get("by_category", {}).get(c, 0) for c in cats]
    ax2.barh(range(len(cats)), vals, color=[PALETTE[2], PALETTE[1], "#c9c8c2"],
             edgecolor=SURFACE, linewidth=1.4, height=0.62)
    for i, v in enumerate(vals):
        ax2.text(v, i, f" {v}", va="center", ha="left", color=INK2, fontsize=7.8)
    ax2.set_yticks(range(len(cats)))
    ax2.set_yticklabels([c.replace("_", " ") for c in cats])
    ax2.set(xlabel="Format C passes", title="Abstention split (PLAN §4.1)")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    save_fig(fig, "fig2_quadrant", cfg,
             "Figure 2 — epistemic alignment quadrants of calibrated verbal vs "
             "behavioral/internal (performative certainty / excessive hedging / grounded "
             "certainty / honest doubt) with the abstention split. Predicted under H2: the "
             "mismatch quadrants cluster by question type across multiple cells. "
             "Null: uniform scatter, no replicated clustering.")


def fig3_model_delta(h3: dict, cfg: Config) -> None:
    """Figure 3 — base vs Instruct: unwarranted stated-confidence rate and
    missed-knowledge rate (neutral terminology per audit reframing)."""
    apply_style(cfg)
    if not h3.get("available"):
        return
    fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH * 0.62, cfg.FIG_WIDTH * 0.42))
    groups = ["unwarranted stated confidence", "missed knowledge"]
    a_name = str(h3.get("model_base", "base"))
    b_name = str(h3.get("model_instruct", "instruct"))
    base = [h3.get("unwarranted_confidence_rate", {}).get("base", np.nan),
            (h3.get("missed_knowledge_rate", {}) or {}).get(a_name, np.nan)]
    inst = [h3.get("unwarranted_confidence_rate", {}).get("instruct", np.nan),
            (h3.get("missed_knowledge_rate", {}) or {}).get(b_name, np.nan)]
    x = np.arange(len(groups))
    w = 0.34
    ax.bar(x - w / 2, base, w * 0.94, color=PALETTE[0], edgecolor=SURFACE, linewidth=1.4, label=f"{a_name} (base)")
    ax.bar(x + w / 2, inst, w * 0.94, color=PALETTE[1], edgecolor=SURFACE, linewidth=1.4, label=f"{b_name} (instruct)")
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
             f"Figure 3 — unwarranted stated-confidence (high verbal / low behavioral-internal) "
             f"and missed-knowledge rates, {a_name} vs {b_name}. Predicted under H3: the "
             f"overstatement rate drops without a matched rise in missed knowledge. "
             f"Null: overlapping bars / equal rates.")


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
                title=f"{re.sub(r'qwen[\\d.]+-', '', model).replace('-instruct', '')}"
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
    """Supplementary — the ragged grid made visible: pilot accuracy per cell
    with the 25–80% commitment band marked."""
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
            ok = commitments.get(cid, {}).get("committed", False)
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.4,
                    color=SURFACE if M[i, j] > 0.55 else INK,
                    fontweight="bold" if ok else "normal")
            if not ok:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=PALETTE[7], lw=1.6, ls=":"))
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"{t}\n{TIER_SPECS[t]['family'][:4]}" for t in tiers])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([re.sub(r"qwen[\d.]+-", "", m) for m in models])
    ax.set_title(f"Pilot accuracy per cell (all committed; shading = outside "
                 f"{cfg.ACCURACY_BAND[0]:.0%}–{cfg.ACCURACY_BAND[1]:.0%} band, kept for matching)")
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, label="pilot accuracy", fraction=0.035, pad=0.02)
    fig.tight_layout()
    save_fig(fig, "fig5_cell_commitment_grid", cfg,
             "Supplementary — pilot accuracy per (model × tier) cell. A ragged grid is the "
             "planned outcome (PLAN §3, §10), not a failure.")


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
    ax.set_yticklabels([f"{re.sub(r'qwen[\\d.]+-', '', m)}·{t}" for m, t in piv.index], fontsize=6.8)
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
        dict(hypothesis="H2", gate="Gate 1+3", passed=h2.get("h2_pass"), verdict=h2.get("verdict")),
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


def verify_model_specs(cfg: Config) -> dict:
    """Audit task (Model Family Configuration), AUTOMATED: download each active
    checkpoint's config.json (cheap, no weights), read the TEXT stack's
    num_hidden_layers / hidden_size, and compare with MODEL_SPECS. Any drift is
    corrected IN PLACE and logged loudly — percentile_layers() taps layer
    indices derived from `layers`, so a stale value would silently probe the
    wrong depths. Fails hard only when a config cannot be fetched at all."""
    res = {}
    for name in cfg.active_models():
        spec = MODEL_SPECS[name]
        entry = {"hf_id": spec["hf_id"]}
        try:
            ac = AutoConfig.from_pretrained(spec["hf_id"], trust_remote_code=False)
            tc = getattr(ac, "text_config", ac)
            real_layers = int(getattr(tc, "num_hidden_layers", -1))
            real_hidden = int(getattr(tc, "hidden_size", -1))
            entry.update(real_layers=real_layers, real_hidden=real_hidden,
                         spec_layers=spec["layers"], spec_hidden=spec["hidden"])
            drift = []
            if real_layers > 0 and spec["layers"] != real_layers:
                LOG.log("model_spec_corrected", model=name, field="layers",
                        was=spec["layers"], now=real_layers)
                spec["layers"] = real_layers
                drift.append("layers")
            if real_hidden > 0 and spec["hidden"] != real_hidden:
                LOG.log("model_spec_corrected", model=name, field="hidden",
                        was=spec["hidden"], now=real_hidden)
                spec["hidden"] = real_hidden
                drift.append("hidden")
            entry["drift_corrected"] = drift
            entry["ok"] = True
            LOG.log("model_spec_verified" if not drift else "model_spec_drift_fixed",
                    model=name, layers=real_layers, hidden=real_hidden)
        except Exception as exc:                           # noqa: BLE001
            entry["ok"] = False
            entry["error"] = str(exc)[:200]
            LOG.log("model_spec_verify_failed", model=name, error=entry["error"])
        res[name] = entry
    bad = [k for k, v in res.items() if not v.get("ok")]
    json_write(PATHS["derived"] / "model_spec_verification.json", res)
    if bad:
        raise RuntimeError(
            f"could not fetch configs for: {bad} — fix hf_ids or network before running; "
            f"tapping activations against unverified layer counts risks silent wrong-layer probes")
    return res


def validate_chat_templates(cfg: Config) -> dict:
    """Audit task (Model Family Configuration): render every ladder rung's chat
    template ONCE before any generation, verifying (a) apply_chat_template
    works, (b) every instruct rung ends with the same assistant trigger so the
    prompt-ending token — which activations tap — is consistent, and (c) base
    models are flagged few-shot. Cheap: tokenizer-only, no weights."""
    res = {}
    endings = {}
    for name in cfg.active_models():
        spec = MODEL_SPECS[name]
        entry = {"hf_id": spec["hf_id"], "chat": bool(spec["chat"])}
        try:
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], trust_remote_code=False)
            if spec["chat"]:
                msgs = [{"role": "system", "content": "t"}, {"role": "user", "content": "q"}]
                rendered = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **think_off_kwargs(tok))
                rendered = normalize_assistant_tail(rendered)
                entry["renders"] = True
                entry["endswith"] = rendered[-24:]
                endings[name] = rendered[-24:]
            else:
                entry["renders"] = None
                entry["note"] = "base model — few-shot prompts, no chat template"
            LOG.log("chat_template_ok" if (not spec["chat"] or entry.get("renders"))
                    else "chat_template_failed", model=name)
        except Exception as exc:                           # noqa: BLE001
            entry["renders"] = False
            entry["error"] = str(exc)[:200]
            LOG.log("chat_template_failed", model=name, error=entry["error"])
        res[name] = entry
    distinct = set(endings.values())
    if len(distinct) > 1:
        LOG.log("prompt_ending_inconsistent", endings=endings,
                note="instruct rungs end with DIFFERENT assistant triggers — "
                     "the tapped prompt-ending token would not be constant")
    res["_consistent_prompt_ending"] = len(distinct) <= 1
    json_write(PATHS["derived"] / "chat_template_validation.json",
               {k: v for k, v in res.items()})
    return res


def stage_report(bank: dict, commitments: dict, h0: dict, h1: dict, h2: dict, h3: dict,
                 h4: dict, gate3: dict, judge: dict, sanity: dict, hlr: dict,
                 manual_sheet: dict, cfg: Config) -> dict:
    ledger = compute_ledger(cfg)
    total_h = float(ledger["gpu_hours"].sum()) if len(ledger) else 0.0
    committed = [k for k, v in commitments.items() if v["committed"]]
    # Gate 1 closes only with BOTH opinions: automated judge agreement AND the
    # human check sheet (audit fix — Gate 1 was `null`/assumed-passed while the
    # sheet sat unfilled).
    auto_ok = judge.get("gate1_pass")
    manual_ok = manual_sheet.get("gate1_manual_pass") if manual_sheet.get("available") else None
    gate1_combined = None if (auto_ok is None and manual_ok is None) \
        else bool(auto_ok is not False and manual_ok is not False)
    gates = {
        "gate1_grading_sanity": {
            "pass": gate1_combined,
            "automated_agreement": judge.get("agreement_overall"),
            "manual_agreement": manual_sheet.get("agreement"),
            "manual_filled": f"{manual_sheet.get('n_filled', 0)}/{manual_sheet.get('n_total', 0)}"
                             if manual_sheet.get("available") else "0/0",
            "note": ("Requires BOTH the judge agreement and the human check sheet to clear "
                     "the threshold (PLAN §16)."),
        },
        "gate2_format_agreement": {"pass": h0.get("gate2_pass"),
                                   "min_lower_ci": min([v["lo"] for v in h0.get("pairs", {}).values()
                                                        if np.isfinite(v.get("lo", np.nan))] or [np.nan])},
        "gate3_probe_validity": {"cells": len(gate3), "passed": sum(v["passes"] for v in gate3.values()),
                                 "dirty_activation_cells": sum(1 for v in gate3.values()
                                                               if not v["activations_clean"])},
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
        "entropy_sanity": sanity,
        "hierarchical_regression": {k: hlr.get(k) for k in ("method", "n", "pseudo_r2")},
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

    # §17.2 run-log row, ready to paste into PLAN.md — now carries full
    # provenance (audit fix: the log was empty and runs recorded "nogit").
    lines = ["| Run-log ID | What it did | Outcome | Headline |", "|---|---|---|---|"]
    prov_line = (f"| {cfg.RUN_NAME}_provenance | code_sha={PROV.get('code_sha')} "
                 f"config={cfg.hash()} platform={cfg.resolved_platform()} seed={cfg.SEED} "
                 f"gpu_h={round(total_h, 2)} cells={len(committed)}/{len(commitments)} "
                 f"finished={report['finished_utc']} | metadata | — |")
    lines.append(prov_line)
    for hid, verdict in report["hypotheses"].items():
        if verdict:
            lines.append(f"| {cfg.RUN_NAME}_{hid} | {hid} per PLAN §13 @ {PROV.get('code_sha')} | "
                         f"{'pass' if 'supported' in str(verdict) else 'null/falsified'} | {verdict} |")
    (PATHS["meta"] / "run_log_rows.md").write_text("\n".join(lines), encoding="utf-8")
    LOG.log("report_written", cells_committed=len(committed), gpu_hours=round(total_h, 2),
            code_sha=PROV.get("code_sha"))
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

    # Provenance-of-architecture: verify layers/hidden against each
    # checkpoint's real config (auto-corrects MODEL_SPECS on drift), then
    # chat-template / prompt-ending validation — all before any GPU compute.
    verify_model_specs(cfg)
    validate_chat_templates(cfg)

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
    gate3 = gate3_verdict(sweep, cfg)

    signals, meta = assemble_signals(bank, graded, entropy, sweep, commitments, cfg) \
        if "calibrate" in cfg.STAGES else (pd.DataFrame(), {"verbal_long": pd.DataFrame()})

    h0 = h1 = h2 = h3 = h4 = hlr = {}
    abst = {"by_category": {}, "per_cell": {}}
    corr, omni = pd.DataFrame(), pd.DataFrame()
    pop, dmatch = {"available": False}, {"available": False}

    # ---- judge audit runs BEFORE the hypothesis tests: H2's restored PLAN §16
    # conjunct needs the Gate 1 verdict ("AND Gate 1 holds") available now.
    judge = stage_judge(bank, graded, judge_cfg, cfg) if judge_cfg.ENABLED else {"enabled": False}

    if "stats" in cfg.STAGES and len(signals):
        feats = question_features(bank)
        h0 = test_h0_format_agreement(meta["verbal_long"], cfg)
        h1 = test_h1_signal_calibration(signals, cfg)
        h2 = test_h2_quadrants(signals, bank, cfg,
                               gate1_pass=judge.get("gate1_pass"), gate3=gate3)
        abst = abstention_split(graded, cfg)
        omni = omniscience_index(graded)
        # Gate 4 usability bar: base-model elicitation must actually parse
        # (audit fix — the guard existed as a config knob but was never wired).
        elic_ok = None
        base_names = [m for m, s in MODEL_SPECS.items() if s.get("rung") == "h3-comparison"]
        if len(graded) and base_names:
            bg = graded[(graded.model.isin(base_names)) & (graded.variant == "FORCED")]
            if len(bg):
                elic_ok = bool(bg["parse_ok"].astype(float).mean()
                               >= cfg.BASE_ELICITATION_MIN_PARSE_RATE)
        h3 = test_h3_base_vs_instruct(signals, abst, cfg, gate4_elicit_ok=elic_ok)
        h4 = test_h4_depth(sweep, cfg)
        hlr = hierarchical_regression(signals, cfg, feats=feats)
        corr = correlation_table(signals, cfg)
        pop = popularity_contrast(signals, cfg)          # recovered R1-vs-R2 gradient
        dmatch = difficulty_matched_view(signals, graded, cfg)

    if len(graded):
        export_manual_check_sheet(bank, graded, cfg)
    manual_sheet = ingest_manual_sheet(cfg)

    if "figures" in cfg.STAGES:
        stage_figures(signals, sweep, h1, h3, abst, corr, commitments, cfg)
    if "tables" in cfg.STAGES:
        stage_tables(bank, graded, signals, sweep, entropy, corr, commitments,
                     h0, h1, h2, h3, h4, abst, omni, cfg)
    report = stage_report(bank, commitments, h0, h1, h2, h3, h4, gate3, judge, sanity,
                          hlr, manual_sheet, cfg) \
        if "report" in cfg.STAGES else {}

    free_cuda()
    if cfg.PURGE_WEIGHTS == "after_run":
        purge_all_weights()
    LOG.log("pipeline_done", minutes=round((time.time() - t_start) / 60, 1))
    return {"bank": bank, "commitments": commitments, "graded": graded, "entropy": entropy,
            "sweep": sweep, "signals": signals, "h0": h0, "h1": h1, "h2": h2, "h3": h3,
            "h4": h4, "hlr": hlr, "corr": corr, "omni": omni, "abstention": abst,
            "popularity": pop, "difficulty_matched": dmatch,
            "gate3": gate3, "judge": judge, "manual_sheet": manual_sheet, "report": report}


# %%
# ============================================================================
# CELL 23 — RUN
# Safe to re-execute: every stage is idempotent and resumes from checkpoints.
# ============================================================================
if __name__ == "__main__":
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
