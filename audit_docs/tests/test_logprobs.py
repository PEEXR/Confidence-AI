"""sequence_logprobs must equal a naive per-token reference on the RAW model
distribution, for real generate() output, under warping and early stopping."""
import ast, sys, types
from pathlib import Path
import numpy as np, torch
from typing import Any, Callable, Sequence
try:
    from transformers import Qwen2Config, Qwen2ForCausalLM
except ImportError:
    print("SKIPPED — transformers not installed"); sys.exit(0)

ROOT = Path(__file__).resolve().parents[2]
tree = ast.parse((ROOT/"confidence_pipeline.py").read_text())
ns = {"torch": torch, "np": np, "Any": Any, "Callable": Callable, "Sequence": Sequence,
      "LoadedModel": object}
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ("sequence_logprobs","_stop_token_ids"):
        exec(compile(ast.Module([n], []), "<x>", "exec"), ns)
seq_lp, stop_ids_fn = ns["sequence_logprobs"], ns["_stop_token_ids"]

torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=128, eos_token_id=2, pad_token_id=0,
                  bos_token_id=1, tie_word_embeddings=True)
model = Qwen2ForCausalLM(cfg).eval()
model.generation_config.repetition_penalty = 1.05
lm = types.SimpleNamespace(model=model)
tokstub = type("T", (), {"pad_token_id": 0, "eos_token_id": 2})()
STOP = stop_ids_fn(tokstub, model)

def naive(seqs, plen, stop):
    """One token at a time, full precision. The definition."""
    out = []
    with torch.no_grad():
        logits = model(input_ids=seqs).logits.double()
    for r in range(seqs.shape[0]):
        tot, cnt = 0.0, 0
        for t in range(plen, seqs.shape[1]):
            tid = int(seqs[r, t])
            if tid in stop: break
            tot += float(torch.log_softmax(logits[r, t-1], -1)[tid]); cnt += 1
        out.append(tot/cnt if cnt else float("nan"))
    return out

fails = []
CASES = [
    ("greedy",            dict(do_sample=False, num_return_sequences=1,  max_new_tokens=8)),
    ("sampled n=10",      dict(do_sample=True,  num_return_sequences=10, max_new_tokens=8,
                               temperature=0.8, top_p=0.95)),
    ("hot, early stops",  dict(do_sample=True,  num_return_sequences=6,  max_new_tokens=30,
                               temperature=2.0, top_p=0.99)),
    ("max_new_tokens=1",  dict(do_sample=True,  num_return_sequences=4,  max_new_tokens=1,
                               temperature=0.9)),
]
for name, kw in CASES:
    torch.manual_seed(11)
    plen, bs = 6, 3
    ids = torch.randint(3, 60, (bs, plen))
    with torch.no_grad():
        seqs = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                              pad_token_id=0, **kw)
    got = seq_lp(lm, seqs, plen, STOP)
    want = naive(seqs, plen, STOP)
    bad = [(g, w) for g, w in zip(got, want)
           if not (np.isnan(g) and np.isnan(w)) and
              (np.isnan(g) != np.isnan(w) or abs(g - w) > 1e-4)]
    stopped = sum(1 for r in range(seqs.shape[0])
                  if any(int(x) in STOP for x in seqs[r, plen:]))
    if bad:
        fails.append(f"{name}: {len(bad)}/{len(got)} rows differ, e.g. {bad[:3]}")
    else:
        print(f"  ok  {name:20s} rows={len(got):3d} early-stopped={stopped}")

# chunking must not change the answer
torch.manual_seed(11)
ids = torch.randint(3, 60, (3, 6))
with torch.no_grad():
    seqs = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                          do_sample=True, num_return_sequences=8, max_new_tokens=20,
                          temperature=1.2, pad_token_id=0)
ref = seq_lp(lm, seqs, 6, STOP, row_chunk=64, tok_chunk=4096)
for rc, tc in [(1, 1), (2, 3), (3, 7), (5, 64), (100, 100)]:
    alt = seq_lp(lm, seqs, 6, STOP, row_chunk=rc, tok_chunk=tc)
    d = [abs(a-b) for a, b in zip(alt, ref) if not (np.isnan(a) or np.isnan(b))]
    if (max(d) if d else 0) > 1e-5 or [np.isnan(x) for x in alt] != [np.isnan(x) for x in ref]:
        fails.append(f"chunking row={rc} tok={tc} changed the result (max diff {max(d):.2e})")
print(f"  ok  chunk invariance   ({len(ref)} rows across 5 chunk settings)")

# ---- LEFT PADDING: the mask and position ids must be honoured ---------
# The tokenizer pads left and generate() keeps that padding, so a plain
# forward would let each row attend to its own pads and number positions from
# 0 instead of from the first real token.
PAD = 0
pids  = torch.tensor([[PAD, PAD, 11, 12, 13], [21, 22, 23, 24, 25]])
pmask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
pgen  = torch.tensor([[31, 32, 33], [41, 42, 43]])
pseq  = torch.cat([pids, pgen], dim=1)
pfull = torch.cat([pmask, torch.ones_like(pgen)], dim=1)

def unpadded_reference(seqs, mask_full, plen, stop):
    """Score each row ALONE, with the padding physically removed."""
    out = []
    for r in range(seqs.shape[0]):
        row = seqs[r][mask_full[r].bool()].unsqueeze(0)
        p_real = int(mask_full[r, :plen].sum())
        with torch.no_grad():
            lg = model(input_ids=row).logits.double()
        tot, cnt = 0.0, 0
        for t in range(p_real, row.shape[1]):
            tid = int(row[0, t])
            if tid in stop: break
            tot += float(torch.log_softmax(lg[0, t-1], -1)[tid]); cnt += 1
        out.append(tot/cnt if cnt else float("nan"))
    return out

want_p = unpadded_reference(pseq, pfull, 5, {2})
got_p = seq_lp(lm, pseq, 5, {2}, attention_mask=pfull)
dp = [abs(a-b) for a, b in zip(got_p, want_p)]
if max(dp) > 1e-4:
    fails.append(f"left padding corrupts scores: diffs {[f'{x:.2e}' for x in dp]} "
                 f"(row 0 is the padded one)")
else:
    print(f"  ok  left padding      max diff {max(dp):.1e} vs unpadded reference")
# omitting the mask on padded input must NOT silently match — proves the mask matters
got_nomask = seq_lp(lm, pseq, 5, {2})
if abs(got_nomask[0] - want_p[0]) < 1e-6:
    fails.append("mask made no difference on a padded row — the test is not exercising it")
# a mismatched mask must raise, not be silently ignored
try:
    seq_lp(lm, pseq, 5, {2}, attention_mask=pfull[:, :-1])
    fails.append("a wrong-shaped attention_mask was accepted")
except ValueError:
    print("  ok  wrong-shaped mask rejected")

# a row that is entirely stop tokens -> NaN, never 0.0
allstop = torch.cat([torch.randint(3, 60, (1, 6)), torch.zeros(1, 5, dtype=torch.long)], 1)
r = seq_lp(lm, allstop, 6, STOP)
if not np.isnan(r[0]):
    fails.append(f"all-stop row returned {r[0]}, must be NaN")
else:
    print("  ok  all-stop row -> NaN")
# no continuation at all
if not np.isnan(seq_lp(lm, torch.randint(3, 60, (2, 6)), 6, STOP)[0]):
    fails.append("empty continuation must be NaN")
else:
    print("  ok  empty continuation -> NaN")

print("\nFAILURES:" if fails else "\nSEQUENCE LOGPROBS EXACT vs NAIVE REFERENCE")
for f in fails: print("  -", f)
sys.exit(1 if fails else 0)
