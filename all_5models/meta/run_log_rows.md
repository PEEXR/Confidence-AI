| Run-log ID | What it did | Outcome | Headline |
|---|---|---|---|
| all_5models_provenance | code_sha=v3-qwen35-audit-fixes config=d8a031a7e859 platform=local seed=20260813 gpu_h=0.0 cells=30/30 finished=2026-08-25T07:12:01+00:00 | metadata | — |
| all_5models_H0 | H0 per PLAN §13 @ v3-qwen35-audit-fixes | null/falsified | H0 falsified — report all three formats separately (PLAN §16 Gate 2 fallback) |
| all_5models_H1 | H1 per PLAN §13 @ v3-qwen35-audit-fixes | null/falsified | H1 falsified on direction — signals differ beyond noise, but the ordering is reversed: verbal is better calibrated than internal. ECE rank (best first): ['verbal', 'behavioral', 'internal']; resolution rank (highest first): ['behavioral', 'verbal', 'internal']. Note low verbal ECE with low resolution indicates a near-constant predictor. |
| all_5models_H2 | H2 per PLAN §13 @ v3-qwen35-audit-fixes | pass | H2 supported — replicated feature association with the alignment taxonomy, with Gate 1 and Gate 3 holding |
| all_5models_H3 | H3 per PLAN §13 @ v3-qwen35-audit-fixes | null/falsified | H3 falsified / null — delta CI includes 0, the missed-knowledge guard fired, or base-model elicitation did not meet the usability bar |
| all_5models_H4 | H4 per PLAN §13 @ v3-qwen35-audit-fixes | null/falsified | H4 falsified / null — onset curves overlap, or reasoning never reaches the gate |