"""Regenerate confidence_pipeline.ipynb from confidence_pipeline.py.

Usage:  python audit_docs/sync_notebook.py


The .py is the git-diffable source; the .ipynb is what actually runs on
molab/Kaggle. Verified round-trip: splitting the pre-edit .py on '# %%'
reproduced all 25 notebook cells byte-for-byte.
"""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
py = (ROOT / "confidence_pipeline.py").read_text()
nb = json.loads((ROOT / "confidence_pipeline.ipynb").read_text())
parts = re.split(r'^# %%\n', py, flags=re.M)

def to_cell(part):
    if part.startswith('# %% [markdown]\n'):
        body = part[len('# %% [markdown]\n'):]
        txt = '\n'.join(l[2:] if l.startswith('# ') else (l[1:] if l == '#' else l)
                        for l in body.split('\n'))
        return 'markdown', txt.strip('\n')
    return 'code', part.strip('\n')

def src_lines(txt):
    lines = txt.split('\n')
    return [l + '\n' for l in lines[:-1]] + [lines[-1]] if lines else []

old_by_i = {i: c for i, c in enumerate(nb['cells'])}
cells = []
for i, part in enumerate(parts):
    kind, txt = to_cell(part)
    prev = old_by_i.get(i)
    if kind == 'markdown':
        cells.append({"cell_type": "markdown", "metadata": (prev or {}).get("metadata", {}),
                      "source": src_lines(txt)})
    else:
        cells.append({"cell_type": "code",
                      "execution_count": None,
                      "metadata": (prev or {}).get("metadata", {}),
                      "outputs": [],
                      "source": src_lines(txt)})
nb['cells'] = cells
rendered = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
target = ROOT / "confidence_pipeline.ipynb"

if "--check" in sys.argv:
    # Gate for CI / run_all.py: fail if the notebook is behind the .py.
    current = target.read_text() if target.exists() else ""
    if current != rendered:
        old_cells = json.loads(current)["cells"] if current else []
        drift = [i for i, (a, b) in enumerate(zip(old_cells, cells))
                 if "".join(a["source"]) != "".join(b["source"])]
        print(f"OUT OF SYNC — confidence_pipeline.ipynb is behind confidence_pipeline.py")
        print(f"  cell count : {len(old_cells)} -> {len(cells)}")
        print(f"  cells differing: {drift or 'none (count changed)'}")
        print(f"  fix: python audit_docs/sync_notebook.py")
        sys.exit(1)
    print(f"in sync: {len(cells)} cells")
    sys.exit(0)

target.write_text(rendered)
print(f"notebook regenerated: {len(cells)} cells "
      f"({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
