# Fair Comparison Results v2 (2026-07-18)

## Scope

This report compares the preserved before run in `results/fair_comparison/` with the v2 run in `results/fair_comparison_v2/`. The v2 run keeps the MemoryBank ranking formula, MethodResult structure, dataset, top_k, candidate_k, token budget, forgetting settings, and no-LLM/no-cloud-API policy unchanged.

## Fixes

- `memorybank_dual_views` now uses annotated query type as an oracle-query-type ablation: CURRENT uses current view, HISTORICAL uses point-in-time history view, CHANGE uses current+history, GENERAL falls back to current view. It is not a deployable method.
- `statebudgetmem_rule` now recognizes Chinese month/stage/history/event-before expressions and uses current-view fallback when a query still falls to GENERAL, avoiding empty candidate sets from routing misses.
- `statebudgetmem_oracle` keeps GENERAL as no personal-memory retrieval.


## Overall Metrics

| method | recall before | recall after | delta recall | valid before | valid after | delta valid | stale before | stale after | delta stale | token before | token after | empty before | empty after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memorybank_core | 0.5427 | 0.5427 | +0.0000 | 0.5391 | 0.5391 | +0.0000 | 0.1701 | 0.1701 | +0.0000 | 32.0000 | 32.0000 | 0.0000 | 0.0000 |
| memorybank_versioning | 0.4394 | 0.4394 | +0.0000 | 0.6970 | 0.6970 | +0.0000 | 0.0035 | 0.0035 | +0.0000 | 28.8750 | 28.8750 | 0.0000 | 0.0000 |
| memorybank_dual_views | 0.5427 | 0.4758 | -0.0668 | 0.5391 | 0.7335 | +0.1944 | 0.1701 | 0.0035 | -0.1667 | 32.0000 | 29.2917 | 0.0000 | 0.0000 |
| statebudgetmem_rule | 0.2506 | 0.4715 | +0.2208 | 0.3455 | 0.7292 | +0.3837 | 0.0035 | 0.0035 | +0.0000 | 14.8438 | 29.1458 | 0.5208 | 0.0000 |
| statebudgetmem_oracle | 0.4758 | 0.4758 | +0.0000 | 0.7335 | 0.7335 | +0.0000 | 0.0035 | 0.0035 | +0.0000 | 29.2917 | 29.2917 | 0.0000 | 0.0000 |

## Query Type Metrics After

### CURRENT

| method | recall | valid recall | stale rate | token proxy | retrieval ms | retrieved | empty rate | predicted counts |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| memorybank_core | 0.5243 | 0.8385 | 0.1667 | 31.8750 | 20.4367 | 3.0000 | 0.0000 | `{"null": 32}` |
| memorybank_versioning | 0.4920 | 0.8750 | 0.0104 | 30.5625 | 23.0703 | 3.0000 | 0.0000 | `{"null": 32}` |
| memorybank_dual_views | 0.4920 | 0.8750 | 0.0104 | 30.5625 | 19.7642 | 3.0000 | 0.0000 | `{"CURRENT": 32}` |
| statebudgetmem_rule | 0.4920 | 0.8750 | 0.0104 | 30.5625 | 20.9033 | 3.0000 | 0.0000 | `{"CURRENT": 28, "GENERAL": 4}` |
| statebudgetmem_oracle | 0.4920 | 0.8750 | 0.0104 | 30.5625 | 19.1616 | 3.0000 | 0.0000 | `{"CURRENT": 32}` |

### HISTORICAL

| method | recall | valid recall | stale rate | token proxy | retrieval ms | retrieved | empty rate | predicted counts |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| memorybank_core | 0.5594 | 0.2344 | 0.3438 | 32.2812 | 20.2298 | 3.0000 | 0.0000 | `{"null": 32}` |
| memorybank_versioning | 0.3911 | 0.7812 | 0.0000 | 25.4688 | 24.0031 | 2.6562 | 0.0000 | `{"null": 32}` |
| memorybank_dual_views | 0.3911 | 0.7812 | 0.0000 | 25.4688 | 20.3711 | 2.6562 | 0.0000 | `{"HISTORICAL": 32}` |
| statebudgetmem_rule | 0.3911 | 0.7812 | 0.0000 | 25.4688 | 21.1021 | 2.6562 | 0.0000 | `{"GENERAL": 1, "HISTORICAL": 31}` |
| statebudgetmem_oracle | 0.3911 | 0.7812 | 0.0000 | 25.4688 | 19.2861 | 2.6562 | 0.0000 | `{"HISTORICAL": 32}` |

### CHANGE

| method | recall | valid recall | stale rate | token proxy | retrieval ms | retrieved | empty rate | predicted counts |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| memorybank_core | 0.5443 | 0.5443 | 0.0000 | 31.8438 | 21.9893 | 3.0000 | 0.0000 | `{"null": 32}` |
| memorybank_versioning | 0.4349 | 0.4349 | 0.0000 | 30.5938 | 23.4984 | 3.0000 | 0.0000 | `{"null": 32}` |
| memorybank_dual_views | 0.5443 | 0.5443 | 0.0000 | 31.8438 | 21.7448 | 3.0000 | 0.0000 | `{"CHANGE": 32}` |
| statebudgetmem_rule | 0.5312 | 0.5312 | 0.0000 | 31.4062 | 21.3025 | 3.0000 | 0.0000 | `{"CHANGE": 19, "CURRENT": 1, "GENERAL": 12}` |
| statebudgetmem_oracle | 0.5443 | 0.5443 | 0.0000 | 31.8438 | 20.1747 | 3.0000 | 0.0000 | `{"CHANGE": 32}` |

## Diagnostic Checks

- core vs dual_views identical retrieved IDs before: 96/96
- core vs dual_views identical retrieved IDs after: 42/96
- statebudgetmem_rule routing before: `{"CHANGE->CHANGE": 19, "CHANGE->CURRENT": 1, "CHANGE->GENERAL": 12, "CURRENT->CURRENT": 25, "CURRENT->GENERAL": 7, "HISTORICAL->GENERAL": 31, "HISTORICAL->HISTORICAL": 1}`
- statebudgetmem_rule routing after: `{"CHANGE->CHANGE": 19, "CHANGE->CURRENT": 1, "CHANGE->GENERAL": 12, "CURRENT->CURRENT": 28, "CURRENT->GENERAL": 4, "HISTORICAL->GENERAL": 1, "HISTORICAL->HISTORICAL": 31}`
- gold/view mismatch cases: 5; see `gold_view_mismatch_cases.json`.

## Interpretation

- `memorybank_dual_views` is no longer equivalent to `memorybank_core`; it should be reported as an oracle-query-type views ablation.
- `statebudgetmem_rule` now tracks the oracle/dual-view ablation closely on this dataset, so the previous low HISTORICAL recall was mostly an implementation/routing coverage issue.
- The stale-retrieval conclusion is stronger after v2: version/view-aware variants keep stale retrieval near zero while improving valid recall over flat MemoryBank core.
- Five HISTORICAL cases still show gold/view semantic mismatch and should be audited before treating point-in-time view results as final benchmark truth.
