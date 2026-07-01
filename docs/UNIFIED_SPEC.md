# Unified Development Interface v0.1

## Common Records

All comparable modules use the existing `MemoryRecord`, `QueryRecord`, and `Scenario` schemas. Do not define parallel memory, query, or scenario records for new methods.

## Method Interface

All comparable methods implement:

- `reset()`: clear state from the previous scenario.
- `ingest(memories)`: load the `MemoryRecord` objects for one scenario.
- `retrieve(query, *, top_k, token_budget=None, mutate=False)`: retrieve memories for one `QueryRecord`.

## Method Output

All methods return `MethodResult`. Returned memories are represented as `RetrievedMemory`, with ranks starting at 1 and `total_token_cost` equal to the sum of returned memory token costs. Method outputs must not include gold fields such as valid or stale labels.

## Fair Comparison

Comparative experiments use the same data, Top-K, token budget, and model configuration for every method being compared.

## Public Metrics

The shared retrieval metrics are Recall@K, Valid Recall@K, Stale Retrieval Rate, total token cost, and retrieval latency.

## Modules

- `baselines`: basic memory and comparison methods.
- `versioning`: state version management.
- `views`: current/history dual views.
- `routing`: query classification and routing.
