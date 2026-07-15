# Contract Test Documentation (费哲瀚 — 成员 D)

## Quick start

```bash
# Run all contract tests (6 methods × 10 tests = 60)
pytest tests/core/test_method_contracts.py -v

# Run only failing tests with verbose output
pytest tests/core/test_method_contracts.py -v --tb=long

# Skip slow methods (for quick iteration)
pytest tests/core/test_method_contracts.py -k "tfidf_topk" -v
```

## Test Matrix

Each test is parameterized over all registered methods. Current coverage:

| Test | What it checks | Failure means |
|------|---------------|---------------|
| `test_reset_clears_state` | `reset()` clears all memory — same query gives same result | Adapter leaks state across scenarios |
| `test_empty_ingest_is_safe` | `ingest([])` does not crash | Adapter cannot handle empty input |
| `test_top_k_bound` | Returned count ≤ `top_k` | Adapter returns more memories than requested |
| `test_total_token_cost_equals_sum` | `total_token_cost` = sum of per-memory costs | Schema inconsistency |
| `test_token_budget_enforced` | Result does not exceed `token_budget` | Budget constraint violated |
| `test_no_gold_leakage_in_result` | `MethodResult` does not expose gold fields | Information leak to retrieval |
| `test_independent_queries_are_isolated` | Query order does not affect results in `independent` mode | Reinforcement/caching leaks between queries |
| `test_methodresult_schema_valid` | `MethodResult` passes Pydantic validation | Broken schema |
| `test_latency_is_end_to_end` | `latency_ms > 0` | Adapter does not measure latency |
| `test_result_contains_required_metadata_keys` | Metadata includes `source_retriever` or `candidate_k` | Missing provenance info |

## When a test fails

1. Look at the method name in the failure: `test_xxx[memorybank_versioning]` tells you which method failed
2. Check the error trace for `InvalidDecisionError`, `ValueError`, or `AssertionError`
3. Run the specific test with verbose output:
   ```bash
   pytest tests/core/test_method_contracts.py::test_reset_clears_state -k "versioning" -v --tb=long
   ```

## Resource Summary

```bash
python tools/phase1/resource_summary.py results/interface_smoke
```

Outputs a table with ingest_ms, retrieve_ms, and token_avg per method.

## Method Comparison

```bash
# Terminal table
python tools/phase1/compare_methods.py results/contract_compare --summary

# JSON output
python tools/phase1/compare_methods.py results/contract_compare --json
```
