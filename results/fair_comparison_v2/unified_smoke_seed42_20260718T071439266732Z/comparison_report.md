# Unified Fair Comparison Report: unified_smoke_seed42_20260718T071439266732Z

## Purpose

Compare MemoryBank Core with StateBudgetMem variants under the same dataset, dense backend, retrieval limits, token budget, forgetting settings, reinforcement policy, and query-state policy.

## Fairness Controls

- Dataset: `data\controlled\temporal_challenge_v1.jsonl`
- Methods: memorybank_core, memorybank_versioning, memorybank_dual_views, statebudgetmem_rule, statebudgetmem_oracle
- Embedding backend/model: `sentence-transformers` / `sentence-transformers/all-MiniLM-L6-v2`
- Dense backend: MemoryBank FAISS IndexFlatIP for all MemoryBank-derived methods
- top_k / candidate_k / token_budget: 3 / 20 / 256
- query_state_policy: `independent`
- reinforcement_enabled: `False`
- forgetting_enabled / exclude_forgotten: `True` / `False`
- random_seed: 42

## Method Summary

| method | recall@k | valid_recall@k | stale_rate | token_proxy | retrieval_ms | eligible | candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| memorybank_core | 0.5427 | 0.5391 | 0.1701 | 32.00 | 20.89 | 6.03 | 6.03 |
| memorybank_versioning | 0.4394 | 0.6970 | 0.0035 | 28.88 | 23.52 | 3.76 | 3.76 |
| memorybank_dual_views | 0.4758 | 0.7335 | 0.0035 | 29.29 | 20.63 | 4.39 | 4.39 |
| statebudgetmem_rule | 0.4715 | 0.7292 | 0.0035 | 29.15 | 21.10 | 4.12 | 4.12 |
| statebudgetmem_oracle | 0.4758 | 0.7335 | 0.0035 | 29.29 | 19.54 | 4.39 | 4.39 |

## Oracle Note

`statebudgetmem_oracle` uses annotated query type as an upper-bound routing reference. It is not a deployable method and should not be reported as the final StateBudgetMem result.

## Provenance

- Git commit: `e62cc69e3da70c026d1b9a03511ded7f5071ddbe`
- Dirty worktree: `True`
- Dataset SHA256: `f93331a2d93588fa8931efb4484fce577f5a5c9c4e679c51d4bb0192af6c8dd9`
- Python/platform: `3.13.5` / `Windows-11-10.0.26200-SP0`
