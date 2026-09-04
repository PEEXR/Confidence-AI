### Run `prod500` — 2026-09-04T01:35:57+00:00

- code_sha    : `UNBOUND-nogit`  **UNBOUND — this run is not traceable to a commit**
- config_hash : `b11fe302abde`
- seed        : 20260813
- platform    : molab  ·  dtype torch.bfloat16
- torch 2.11.0+cu130 · transformers 5.14.1 · math_verify True
- grid        : 30/30 cells committed
- GPU-hours   : 4.778
- canonical verbal format: Bfix

| Run-log ID | What it did | Outcome | Headline |
|---|---|---|---|
| prod500_H0 | H0 per PLAN §13 | null/falsified | H0 falsified — report all three formats separately (PLAN §16 Gate 2 fallback) |
| prod500_H1 | H1 per PLAN §13 | null/falsified | H1 falsified — CIs overlap, no ordering distinguishable |
| prod500_H2 | H2 per PLAN §13 | pass | H2 not supported - failing conjunct(s): replicates_across_cells. Reported as a DESCRIPTIVE result: confidence disagreement is heterogeneous across model x tier cells, and the features that survive pooling are tier proxies rather than question characteristics. |
| prod500_H3 | H3 per PLAN §13 | null/falsified | H3 untested — only 0/0 matched rows have BOTH a verbal and a behavioral/internal value. The pre-repair code would have reported a rate of 0.0 here rather than declining to test. |
| prod500_H4 | H4 per PLAN §13 | pass | H4 supported — internal signal onsets later for reasoning than retrieval |