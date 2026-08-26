"""ActivationTap must capture the LAST PROMPT token, under either padding
side, and must not be overwritten by decoding.

Sequence log-probabilities are covered separately in test_logprobs.py, which
needs a real transformers model."""
from pathlib import Path
import ast, sys, types
import numpy as np, torch, torch.nn as nn
from typing import Any, Callable, Sequence

SRC = open(Path(__file__).resolve().parents[2] / "confidence_pipeline.py").read()
tree = ast.parse(SRC)
WANT = {"percentile_layers", "_transformer_base", "last_prompt_index",
        "ActivationTap", "prefill_capture"}
ns = {"torch": torch, "np": np, "Any": Any, "Callable": Callable,
      "Sequence": Sequence, "LoadedModel": object}
picked = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANT:
        exec(compile(ast.Module([node], []), "<x>", "exec"), ns)
        picked.append(node.name)
assert WANT <= set(picked), f"missing {WANT - set(picked)}"
ActivationTap, prefill_capture = ns["ActivationTap"], ns["prefill_capture"]
last_prompt_index = ns["last_prompt_index"]

# ---- fake Qwen2-shaped model -------------------------------------------
H, V, NL = 16, 40, 4
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.lin = nn.Linear(H, H)
    def forward(s, x, **kw):
        return (s.lin(torch.tanh(x)),)          # tuple output, like a real decoder block
class Body(nn.Module):
    def __init__(s):
        super().__init__()
        s.embed_tokens = nn.Embedding(V, H)
        s.layers = nn.ModuleList(Block() for _ in range(NL))
    def forward(s, input_ids=None, attention_mask=None, use_cache=False, **kw):
        h = s.embed_tokens(input_ids)
        for b in s.layers:
            h = b(h)[0]
        return types.SimpleNamespace(last_hidden_state=h)
class FakeLM(nn.Module):
    def __init__(s):
        super().__init__(); s.model = Body(); s.lm_head = nn.Linear(H, V)
torch.manual_seed(0)
fake = FakeLM().eval()
lm = types.SimpleNamespace(name="fake", model=fake, spec={"layers": NL},
                           hidden_device=torch.device("cpu"))

def manual_states(ids):
    """Ground truth: hidden state at each depth, computed by hand."""
    out, h = {}, fake.model.embed_tokens(ids)
    out[0] = h
    for i, b in enumerate(fake.model.layers, start=1):
        h = b(h)[0]; out[i] = h
    return out

PCTS = (0, 25, 50, 75, 100)
fails = []
for side in ("left", "right"):
    if side == "left":
        ids  = torch.tensor([[0, 0, 5, 6, 7], [0, 8, 9, 10, 11]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
        want_idx = [4, 4]
    else:
        ids  = torch.tensor([[5, 6, 7, 0, 0], [8, 9, 10, 11, 0]])
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        want_idx = [2, 3]
    got_idx = last_prompt_index(mask).tolist()
    if got_idx != want_idx:
        fails.append(f"{side}: last_prompt_index {got_idx} != {want_idx}")

    tap = ActivationTap(lm, PCTS)
    with torch.no_grad():
        acts = prefill_capture(lm, {"input_ids": ids, "attention_mask": mask}, tap)
        truth = manual_states(ids)
    pmap = ns["percentile_layers"](NL, PCTS)
    for pct, blk in pmap.items():
        ref = np.stack([truth[blk][r, want_idx[r], :].numpy() for r in range(ids.shape[0])])
        if not np.allclose(acts[pct], ref, atol=1e-5):
            fails.append(f"{side}: p{pct} mismatch (max diff {np.abs(acts[pct]-ref).max():.2e})")

    # ---- simulate decoding: hooks fire again on NEW tokens --------------
    frozen_before = {k: v.copy() for k, v in acts.items()}
    with torch.no_grad():
        for step in range(3):                       # 3 decode steps
            fake.model(input_ids=torch.tensor([[20+step],[21+step]]),
                       attention_mask=torch.ones(2, 1, dtype=torch.long))
    after = tap.pop()
    if after:
        fails.append(f"{side}: tap captured {sorted(after)} DURING decode — not frozen")
    for k in frozen_before:
        if not np.allclose(frozen_before[k], acts[k]):
            fails.append(f"{side}: p{k} mutated after decode")
    tap.close()

# ---- the audit's smoking gun: p0 must be constant on a shared suffix ----
tap = ActivationTap(lm, (0,))
ids  = torch.tensor([[1, 2, 3, 9], [4, 5, 6, 9], [7, 8, 2, 9]])   # same final token
mask = torch.ones_like(ids)
with torch.no_grad():
    a = prefill_capture(lm, {"input_ids": ids, "attention_mask": mask}, tap)
spread = float(np.abs(a[0] - a[0][0]).max())
if spread > 1e-6:
    fails.append(f"p0 varies ({spread:.2e}) across rows sharing a final token — "
                 f"embeddings are position-free, so this must be 0")
tap.close()

# ---- a stale MODEL_SPECS['layers'] must RAISE, not silently re-scale ----
for stale in (NL - 1, NL + 1):                 # under- and over-count
    try:
        t = ActivationTap(types.SimpleNamespace(name="stale", model=fake,
                                                spec={"layers": stale}), PCTS)
        t.close()
        fails.append(f"spec layers={stale} (real {NL}) did NOT raise — depths mis-map silently")
    except RuntimeError:
        pass
# and no hooks may survive a failed construction
before = sum(len(getattr(m, "_forward_hooks", {})) for m in fake.modules())
try:
    ActivationTap(types.SimpleNamespace(name="stale", model=fake, spec={"layers": NL + 1}), PCTS)
except RuntimeError:
    pass
after = sum(len(getattr(m, "_forward_hooks", {})) for m in fake.modules())
if after != before:
    fails.append(f"failed ActivationTap leaked {after - before} hooks")

# ---- wrong architecture must raise, not guess --------------------------
try:
    ActivationTap(types.SimpleNamespace(name="x", model=nn.Linear(4, 4), spec={"layers": 2}), (0,))
    fails.append("no-transformer-body model did NOT raise")
except RuntimeError:
    pass

print("FAILURES:" if fails else "ALL TAP TESTS PASSED")
for f in fails: print("  -", f)
sys.exit(1 if fails else 0)
