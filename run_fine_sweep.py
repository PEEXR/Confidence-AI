"""Run the isolated 5% fine probe sweep on Molab / cloud."""
from dataclasses import replace
import sys
from pathlib import Path

# Ensure repo root is in python path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from confidence_pipeline import CFG, JUDGE, run_pipeline

FINE_CFG = replace(
    CFG,
    RUN_NAME="prod500_5pct",
    PERCENTILES=tuple(range(0, 101, 5)),
    H4_ONSET_RULE="two_consecutive",
    STAGES=("extract", "probe", "calibrate", "stats", "figures", "tables", "report"),
)

if __name__ == "__main__":
    print(f"=== Starting 5% Fine Probe Sweep ({FINE_CFG.RUN_NAME}) ===")
    print(f"Config hash    : {FINE_CFG.hash()}")
    print(f"Percentiles (21): {FINE_CFG.PERCENTILES}")
    print(f"H4 Onset Rule  : {FINE_CFG.H4_ONSET_RULE}")
    print(f"Stages         : {FINE_CFG.STAGES}")
    RESULTS = run_pipeline(FINE_CFG, JUDGE)
    print("\n=== Fine Probe Sweep Finished Successfully ===")
